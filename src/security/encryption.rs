use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    pub algorithm: EncryptionAlgorithm,
    pub key_size: usize,
    pub iv_size: usize,
    pub salt_size: usize,
    pub iterations: u32,
    pub enable_compression: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    Aes256Gcm,
    ChaCha20Poly1305,
    Aes256Cbc,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    pub algorithm: EncryptionAlgorithm,
    pub salt: Vec<u8>,
    pub iv: Vec<u8>,
    pub data: Vec<u8>,
    pub tag: Option<Vec<u8>>,
    pub checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyDerivationParams {
    pub salt: Vec<u8>,
    pub iterations: u32,
    pub key_length: usize,
}

pub struct Encryptor {
    config: EncryptionConfig,
    master_key: Vec<u8>,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            algorithm: EncryptionAlgorithm::Aes256Gcm,
            key_size: 32, // 256 bits
            iv_size: 12,  // 96 bits for GCM
            salt_size: 16, // 128 bits
            iterations: 100_000,
            enable_compression: false,
        }
    }
}

impl Encryptor {
    pub fn new() -> Result<Self> {
        Self::with_config(EncryptionConfig::default())
    }

    pub fn with_config(config: EncryptionConfig) -> Result<Self> {
        let master_key = Self::generate_master_key()?;
        
        Ok(Self {
            config,
            master_key,
        })
    }

    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        let encrypted_data = self.encrypt_with_password(data, &self.master_key)?;
        Ok(bincode::serialize(&encrypted_data)?)
    }

    pub fn decrypt(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        let encrypted_data: EncryptedData = bincode::deserialize(encrypted_data)?;
        self.decrypt_with_password(&encrypted_data, &self.master_key)
    }

    pub fn encrypt_with_password(&self, data: &[u8], password: &[u8]) -> Result<EncryptedData> {
        // Generate salt and IV
        let salt = self.generate_random_bytes(self.config.salt_size)?;
        let iv = self.generate_random_bytes(self.config.iv_size)?;

        // Derive key from password
        let key = self.derive_key(password, &salt, self.config.iterations, self.config.key_size)?;

        // Compress data if enabled
        let data_to_encrypt = if self.config.enable_compression {
            self.compress_data(data)?
        } else {
            data.to_vec()
        };

        // Encrypt data
        let (encrypted_data, tag) = match self.config.algorithm {
            EncryptionAlgorithm::Aes256Gcm => self.encrypt_aes_gcm(&data_to_encrypt, &key, &iv)?,
            EncryptionAlgorithm::ChaCha20Poly1305 => self.encrypt_chacha20_poly1305(&data_to_encrypt, &key, &iv)?,
            EncryptionAlgorithm::Aes256Cbc => {
                let encrypted = self.encrypt_aes_cbc(&data_to_encrypt, &key, &iv)?;
                (encrypted, None)
            }
        };

        // Calculate checksum
        let checksum = self.calculate_checksum(&encrypted_data);

        Ok(EncryptedData {
            algorithm: self.config.algorithm.clone(),
            salt,
            iv,
            data: encrypted_data,
            tag,
            checksum,
        })
    }

    pub fn decrypt_with_password(&self, encrypted_data: &EncryptedData, password: &[u8]) -> Result<Vec<u8>> {
        // Verify checksum
        let calculated_checksum = self.calculate_checksum(&encrypted_data.data);
        if calculated_checksum != encrypted_data.checksum {
            return Err(crate::error::AppError::SecurityViolation("Data integrity check failed".to_string()));
        }

        // Derive key from password
        let key = self.derive_key(password, &encrypted_data.salt, self.config.iterations, self.config.key_size)?;

        // Decrypt data
        let decrypted_data = match encrypted_data.algorithm {
            EncryptionAlgorithm::Aes256Gcm => {
                let tag = encrypted_data.tag.as_ref().ok_or_else(|| {
                    crate::error::AppError::SecurityViolation("Missing authentication tag".to_string())
                })?;
                self.decrypt_aes_gcm(&encrypted_data.data, &key, &encrypted_data.iv, tag)?
            }
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                let tag = encrypted_data.tag.as_ref().ok_or_else(|| {
                    crate::error::AppError::SecurityViolation("Missing authentication tag".to_string())
                })?;
                self.decrypt_chacha20_poly1305(&encrypted_data.data, &key, &encrypted_data.iv, tag)?
            }
            EncryptionAlgorithm::Aes256Cbc => {
                self.decrypt_aes_cbc(&encrypted_data.data, &key, &encrypted_data.iv)?
            }
        };

        // Decompress data if it was compressed
        let final_data = if self.config.enable_compression {
            self.decompress_data(&decrypted_data)?
        } else {
            decrypted_data
        };

        Ok(final_data)
    }

    pub fn encrypt_file(&self, input_path: &str, output_path: &str) -> Result<()> {
        let data = std::fs::read(input_path)?;
        let encrypted_data = self.encrypt(&data)?;
        std::fs::write(output_path, encrypted_data)?;
        Ok(())
    }

    pub fn decrypt_file(&self, input_path: &str, output_path: &str) -> Result<()> {
        let encrypted_data = std::fs::read(input_path)?;
        let decrypted_data = self.decrypt(&encrypted_data)?;
        std::fs::write(output_path, decrypted_data)?;
        Ok(())
    }

    pub fn encrypt_stream<R: Read, W: Write>(&self, reader: &mut R, writer: &mut W) -> Result<()> {
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;
        let encrypted_data = self.encrypt(&buffer)?;
        writer.write_all(&encrypted_data)?;
        Ok(())
    }

    pub fn decrypt_stream<R: Read, W: Write>(&self, reader: &mut R, writer: &mut W) -> Result<()> {
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;
        let decrypted_data = self.decrypt(&buffer)?;
        writer.write_all(&decrypted_data)?;
        Ok(())
    }

    pub fn generate_key_pair(&self) -> Result<(Vec<u8>, Vec<u8>)> {
        // Generate a new key pair for asymmetric encryption
        // This is a simplified implementation - in production, use proper key generation
        let private_key = self.generate_random_bytes(32)?;
        let public_key = self.derive_key(&private_key, b"public_key_salt", 1, 32)?;
        Ok((private_key, public_key))
    }

    pub fn hash_password(&self, password: &str, salt: &[u8]) -> Result<Vec<u8>> {
        self.derive_key(password.as_bytes(), salt, self.config.iterations, 32)
    }

    pub fn verify_password(&self, password: &str, salt: &[u8], hash: &[u8]) -> Result<bool> {
        let computed_hash = self.hash_password(password, salt)?;
        Ok(self.constant_time_compare(&computed_hash, hash))
    }

    pub fn generate_salt(&self) -> Result<Vec<u8>> {
        self.generate_random_bytes(self.config.salt_size)
    }

    pub fn secure_wipe(&self, data: &mut [u8]) {
        // Overwrite sensitive data with random bytes
        use rand::RngCore;
        let mut rng = rand::thread_rng();
        rng.fill_bytes(data);
    }

    fn generate_master_key() -> Result<Vec<u8>> {
        // In production, this should load from secure storage or environment
        // For now, generate a random key
        use rand::RngCore;
        let mut key = vec![0u8; 32];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut key);
        Ok(key)
    }

    fn derive_key(&self, password: &[u8], salt: &[u8], iterations: u32, key_length: usize) -> Result<Vec<u8>> {
        use sha2::Sha256;
        
        // PBKDF2 key derivation
        let mut key = vec![0u8; key_length];
        pbkdf2::pbkdf2::<hmac::Hmac<Sha256>>(password, salt, iterations, &mut key);
        Ok(key)
    }

    fn generate_random_bytes(&self, length: usize) -> Result<Vec<u8>> {
        use rand::RngCore;
        let mut bytes = vec![0u8; length];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut bytes);
        Ok(bytes)
    }

    fn encrypt_aes_gcm(&self, data: &[u8], key: &[u8], iv: &[u8]) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
        use aes_gcm::{Aes256Gcm, Key, Nonce};
        use aes_gcm::aead::{Aead, NewAead};
        
        let cipher = Aes256Gcm::new(Key::from_slice(key));
        let nonce = Nonce::from_slice(iv);
        
        let ciphertext = cipher.encrypt(nonce, data)
            .map_err(|e| crate::error::AppError::SecurityViolation(format!("Encryption failed: {}", e)))?;
        
        // Extract tag from ciphertext (last 16 bytes)
        let tag_len = 16;
        let data_len = ciphertext.len() - tag_len;
        let encrypted_data = ciphertext[..data_len].to_vec();
        let tag = ciphertext[data_len..].to_vec();
        
        Ok((encrypted_data, Some(tag)))
    }

    fn decrypt_aes_gcm(&self, data: &[u8], key: &[u8], iv: &[u8], tag: &[u8]) -> Result<Vec<u8>> {
        use aes_gcm::{Aes256Gcm, Key, Nonce};
        use aes_gcm::aead::{Aead, NewAead};
        
        let cipher = Aes256Gcm::new(Key::from_slice(key));
        let nonce = Nonce::from_slice(iv);
        
        // Combine data and tag
        let mut ciphertext = data.to_vec();
        ciphertext.extend_from_slice(tag);
        
        let plaintext = cipher.decrypt(nonce, ciphertext.as_ref())
            .map_err(|e| crate::error::AppError::SecurityViolation(format!("Decryption failed: {}", e)))?;
        
        Ok(plaintext)
    }

    fn encrypt_chacha20_poly1305(&self, data: &[u8], key: &[u8], iv: &[u8]) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
        use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
        use chacha20poly1305::aead::{Aead, NewAead};
        
        let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
        let nonce = Nonce::from_slice(iv);
        
        let ciphertext = cipher.encrypt(nonce, data)
            .map_err(|e| crate::error::AppError::SecurityViolation(format!("Encryption failed: {}", e)))?;
        
        // Extract tag from ciphertext (last 16 bytes)
        let tag_len = 16;
        let data_len = ciphertext.len() - tag_len;
        let encrypted_data = ciphertext[..data_len].to_vec();
        let tag = ciphertext[data_len..].to_vec();
        
        Ok((encrypted_data, Some(tag)))
    }

    fn decrypt_chacha20_poly1305(&self, data: &[u8], key: &[u8], iv: &[u8], tag: &[u8]) -> Result<Vec<u8>> {
        use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
        use chacha20poly1305::aead::{Aead, NewAead};
        
        let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
        let nonce = Nonce::from_slice(iv);
        
        // Combine data and tag
        let mut ciphertext = data.to_vec();
        ciphertext.extend_from_slice(tag);
        
        let plaintext = cipher.decrypt(nonce, ciphertext.as_ref())
            .map_err(|e| crate::error::AppError::SecurityViolation(format!("Decryption failed: {}", e)))?;
        
        Ok(plaintext)
    }

    fn encrypt_aes_cbc(&self, data: &[u8], key: &[u8], iv: &[u8]) -> Result<Vec<u8>> {
        use aes::Aes256;
        use block_modes::{BlockMode, Cbc};
        use block_modes::block_padding::Pkcs7;
        
        type Aes256Cbc = Cbc<Aes256, Pkcs7>;
        
        let cipher = Aes256Cbc::new_from_slices(key, iv)
            .map_err(|e| crate::error::AppError::SecurityViolation(format!("Cipher creation failed: {}", e)))?;
        
        let ciphertext = cipher.encrypt_vec(data);
        Ok(ciphertext)
    }

    fn decrypt_aes_cbc(&self, data: &[u8], key: &[u8], iv: &[u8]) -> Result<Vec<u8>> {
        use aes::Aes256;
        use block_modes::{BlockMode, Cbc};
        use block_modes::block_padding::Pkcs7;
        
        type Aes256Cbc = Cbc<Aes256, Pkcs7>;
        
        let cipher = Aes256Cbc::new_from_slices(key, iv)
            .map_err(|e| crate::error::AppError::SecurityViolation(format!("Cipher creation failed: {}", e)))?;
        
        let plaintext = cipher.decrypt_vec(data)
            .map_err(|e| crate::error::AppError::SecurityViolation(format!("Decryption failed: {}", e)))?;
        
        Ok(plaintext)
    }

    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        Ok(encoder.finish()?)
    }

    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;
        
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }

    fn calculate_checksum(&self, data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    fn constant_time_compare(&self, a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        
        let mut result = 0u8;
        for (x, y) in a.iter().zip(b.iter()) {
            result |= x ^ y;
        }
        
        result == 0
    }
}

// Key management utilities
pub struct KeyManager {
    keys: std::collections::HashMap<String, Vec<u8>>,
}

impl KeyManager {
    pub fn new() -> Self {
        Self {
            keys: std::collections::HashMap::new(),
        }
    }

    pub fn generate_key(&mut self, key_id: &str, key_size: usize) -> Result<Vec<u8>> {
        use rand::RngCore;
        let mut key = vec![0u8; key_size];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut key);
        
        self.keys.insert(key_id.to_string(), key.clone());
        Ok(key)
    }

    pub fn get_key(&self, key_id: &str) -> Option<&Vec<u8>> {
        self.keys.get(key_id)
    }

    pub fn rotate_key(&mut self, key_id: &str, new_key: Vec<u8>) -> Result<()> {
        self.keys.insert(key_id.to_string(), new_key);
        Ok(())
    }

    pub fn delete_key(&mut self, key_id: &str) -> Result<()> {
        self.keys.remove(key_id);
        Ok(())
    }

    pub fn export_key(&self, key_id: &str, password: &str) -> Result<Vec<u8>> {
        let key = self.keys.get(key_id).ok_or_else(|| {
            crate::error::AppError::SecurityViolation("Key not found".to_string())
        })?;

        let encryptor = Encryptor::new()?;
        let encrypted_key = encryptor.encrypt_with_password(key, password.as_bytes())?;
        Ok(bincode::serialize(&encrypted_key)?)
    }

    pub fn import_key(&mut self, key_id: &str, encrypted_key_data: &[u8], password: &str) -> Result<()> {
        let encrypted_key: EncryptedData = bincode::deserialize(encrypted_key_data)?;
        let encryptor = Encryptor::new()?;
        let key = encryptor.decrypt_with_password(&encrypted_key, password.as_bytes())?;
        
        self.keys.insert(key_id.to_string(), key);
        Ok(())
    }
}

// Secure random number generation
pub struct SecureRandom;

impl SecureRandom {
    pub fn generate_bytes(length: usize) -> Result<Vec<u8>> {
        use rand::RngCore;
        let mut bytes = vec![0u8; length];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut bytes);
        Ok(bytes)
    }

    pub fn generate_string(length: usize) -> Result<String> {
        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let random_string: String = (0..length)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect();
        
        Ok(random_string)
    }

    pub fn generate_uuid() -> String {
        Uuid::new_v4().to_string()
    }
}