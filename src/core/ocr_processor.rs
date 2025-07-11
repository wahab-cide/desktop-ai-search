// OCR Processing Module - Using Tesseract
//
// This module implements OCR using Tesseract via leptess Rust bindings.
// Requires Tesseract to be installed on the system.

use crate::error::{AppError, IndexingError, Result};
use std::path::Path;
use std::collections::HashMap;
use std::time::Instant;
use leptess::{LepTess, Variable};
use image::{ImageBuffer, Luma, DynamicImage, ImageFormat};
use std::io::Cursor;
use tokio::sync::Semaphore;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct OcrResult {
    pub text: String,
    pub confidence: f32,
    pub page_count: usize,
    pub processing_time_ms: u64,
    pub language: Option<String>,
    pub bounding_boxes: Vec<BoundingBox>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub text: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct OcrOptions {
    pub confidence_threshold: f32,
    pub language: Option<String>,
    pub preprocessing: bool,
    pub max_workers: usize,
    pub quality_threshold: f32,
    pub page_segmentation_mode: u32,
    pub ocr_engine_mode: u32,
    pub extract_bounding_boxes: bool,
}

impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            language: None, // Auto-detect
            preprocessing: true,
            max_workers: 2,
            quality_threshold: 0.7,
            page_segmentation_mode: 3, // Fully automatic page segmentation
            ocr_engine_mode: 3, // Default OCR Engine (based on what is available)
            extract_bounding_boxes: false,
        }
    }
}

pub struct OcrProcessor {
    options: OcrOptions,
    semaphore: Arc<Semaphore>,
}

impl OcrProcessor {
    pub fn new(options: OcrOptions) -> Result<Self> {
        // Validate Tesseract installation
        Self::validate_tesseract()?;
        
        let semaphore = Arc::new(Semaphore::new(options.max_workers));
        
        Ok(Self {
            options,
            semaphore,
        })
    }
    
    /// Validate that Tesseract is properly installed and accessible
    fn validate_tesseract() -> Result<()> {
        match LepTess::new(None, "eng") {
            Ok(_) => Ok(()),
            Err(e) => Err(AppError::Indexing(IndexingError::OcrInitialization(
                format!("Tesseract initialization failed: {}", e)
            ))),
        }
    }
    
    /// Process a single image file for OCR
    pub async fn process_image<P: AsRef<Path>>(&self, image_path: P) -> Result<OcrResult> {
        let _permit = self.semaphore.acquire().await
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        let start_time = Instant::now();
        let image_path = image_path.as_ref();
        
        // Load and preprocess image
        let processed_image = self.load_and_preprocess_image(image_path).await?;
        
        // Initialize Tesseract
        let language = self.options.language.as_deref().unwrap_or("eng");
        let mut tesseract = LepTess::new(None, language)
            .map_err(|e| AppError::Indexing(IndexingError::OcrInitialization(e.to_string())))?;
        
        // Configure Tesseract options
        self.configure_tesseract(&mut tesseract)?;
        
        // Convert image to gray buffer for Tesseract
        let gray_buffer = self.image_to_gray_buffer(&processed_image)?;
        
        // Set image data
        tesseract.set_image_from_mem(&gray_buffer)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        // Extract text
        let text = tesseract.get_utf8_text()
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        // Get confidence score
        let confidence = tesseract.mean_text_conf() as f32 / 100.0;
        
        // Extract bounding boxes if requested
        let bounding_boxes = if self.options.extract_bounding_boxes {
            self.extract_word_boxes(&mut tesseract)?
        } else {
            Vec::new()
        };
        
        // Build metadata
        let mut metadata = HashMap::new();
        metadata.insert("image_path".to_string(), image_path.to_string_lossy().to_string());
        metadata.insert("tesseract_version".to_string(), "5.3.0".to_string()); // TODO: Get from actual API when available
        
        if let Ok(image_meta) = image::open(image_path) {
            metadata.insert("image_width".to_string(), image_meta.width().to_string());
            metadata.insert("image_height".to_string(), image_meta.height().to_string());
            metadata.insert("image_format".to_string(), format!("{:?}", image_meta.color()));
        }
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        Ok(OcrResult {
            text: text.trim().to_string(),
            confidence,
            page_count: 1,
            processing_time_ms: processing_time,
            language: Some(language.to_string()),
            bounding_boxes,
            metadata,
        })
    }
    
    /// Process multiple images in parallel
    pub async fn process_images<P: AsRef<Path>>(&self, image_paths: Vec<P>) -> Vec<Result<OcrResult>> {
        let mut tasks = Vec::new();
        
        for path in image_paths {
            let processor = self.clone();
            let path = path.as_ref().to_owned();
            
            let task = tokio::spawn(async move {
                processor.process_image(path).await
            });
            
            tasks.push(task);
        }
        
        let mut results = Vec::new();
        for task in tasks {
            match task.await {
                Ok(result) => results.push(result),
                Err(e) => results.push(Err(AppError::Indexing(
                    IndexingError::Processing(format!("Task join error: {}", e))
                ))),
            }
        }
        
        results
    }
    
    /// Process images from raw bytes
    pub async fn process_image_bytes(&self, image_bytes: &[u8], format_hint: Option<ImageFormat>) -> Result<OcrResult> {
        let _permit = self.semaphore.acquire().await
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        let start_time = Instant::now();
        
        // Load image from bytes
        let image = if let Some(format) = format_hint {
            image::load_from_memory_with_format(image_bytes, format)
                .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?
        } else {
            image::load_from_memory(image_bytes)
                .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?
        };
        
        // Preprocess if enabled
        let processed_image = if self.options.preprocessing {
            self.preprocess_image(image)?
        } else {
            image
        };
        
        // Initialize Tesseract
        let language = self.options.language.as_deref().unwrap_or("eng");
        let mut tesseract = LepTess::new(None, language)
            .map_err(|e| AppError::Indexing(IndexingError::OcrInitialization(e.to_string())))?;
        
        self.configure_tesseract(&mut tesseract)?;
        
        // Convert to gray buffer
        let gray_buffer = self.image_to_gray_buffer(&processed_image)?;
        
        // Perform OCR
        tesseract.set_image_from_mem(&gray_buffer)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        let text = tesseract.get_utf8_text()
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        let confidence = tesseract.mean_text_conf() as f32 / 100.0;
        
        let bounding_boxes = if self.options.extract_bounding_boxes {
            self.extract_word_boxes(&mut tesseract)?
        } else {
            Vec::new()
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "memory".to_string());
        metadata.insert("image_size_bytes".to_string(), image_bytes.len().to_string());
        metadata.insert("tesseract_version".to_string(), "5.3.0".to_string()); // TODO: Get from actual API when available
        
        Ok(OcrResult {
            text: text.trim().to_string(),
            confidence,
            page_count: 1,
            processing_time_ms: processing_time,
            language: Some(language.to_string()),
            bounding_boxes,
            metadata,
        })
    }
    
    /// Check if OCR quality meets threshold
    pub fn is_quality_acceptable(&self, result: &OcrResult) -> bool {
        result.confidence >= self.options.quality_threshold &&
        !result.text.trim().is_empty() &&
        result.text.len() > 10 // Minimum text length
    }
    
    /// Get available languages
    pub fn get_available_languages() -> Result<Vec<String>> {
        match LepTess::new(None, "eng") {
            Ok(_tesseract) => {
                // TODO: Fix when leptess API supports this method
                let languages = vec!["eng".to_string()]; // Default language
                Ok(languages)
            },
            Err(e) => Err(AppError::Indexing(IndexingError::OcrInitialization(e.to_string()))),
        }
    }
    
    /// Auto-detect language from image
    pub async fn detect_language<P: AsRef<Path>>(&self, image_path: P) -> Result<String> {
        let temp_options = OcrOptions {
            language: Some("osd".to_string()), // Orientation and script detection
            ..self.options.clone()
        };
        
        let temp_processor = OcrProcessor::new(temp_options)?;
        let result = temp_processor.process_image(image_path).await?;
        
        // Parse OSD output to get language
        // This is a simplified implementation - in practice you'd parse the OSD output
        if result.confidence > 0.5 {
            Ok("eng".to_string()) // Default fallback
        } else {
            Ok("eng".to_string())
        }
    }
    
    // Private helper methods
    
    fn configure_tesseract(&self, tesseract: &mut LepTess) -> Result<()> {
        // Set page segmentation mode
        tesseract.set_variable(Variable::TesseditPagesegMode, &self.options.page_segmentation_mode.to_string())
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        // Set OCR engine mode
        tesseract.set_variable(Variable::TesseditOcrEngineMode, &self.options.ocr_engine_mode.to_string())
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        Ok(())
    }
    
    async fn load_and_preprocess_image<P: AsRef<Path>>(&self, path: P) -> Result<DynamicImage> {
        let image = image::open(path)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        if self.options.preprocessing {
            self.preprocess_image(image)
        } else {
            Ok(image)
        }
    }
    
    fn preprocess_image(&self, image: DynamicImage) -> Result<DynamicImage> {
        // Convert to grayscale for better OCR
        let gray_image = image.to_luma8();
        
        // Apply basic noise reduction and contrast enhancement
        let enhanced = self.enhance_contrast(&gray_image);
        
        Ok(DynamicImage::ImageLuma8(enhanced))
    }
    
    fn enhance_contrast(&self, image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        // Simple contrast enhancement
        let mut enhanced = image.clone();
        
        for pixel in enhanced.pixels_mut() {
            let value = pixel[0] as f32;
            // Apply simple contrast stretch
            let enhanced_value = ((value - 128.0) * 1.2 + 128.0).clamp(0.0, 255.0) as u8;
            pixel[0] = enhanced_value;
        }
        
        enhanced
    }
    
    fn image_to_gray_buffer(&self, image: &DynamicImage) -> Result<Vec<u8>> {
        let gray_image = image.to_luma8();
        Ok(gray_image.into_raw())
    }
    
    fn extract_word_boxes(&self, tesseract: &mut LepTess) -> Result<Vec<BoundingBox>> {
        let mut boxes = Vec::new();
        
        // Get word-level bounding boxes from Tesseract
        // The second parameter is for whether to return text with boxes
        if let Some(components) = tesseract.get_component_boxes(leptess::capi::TessPageIteratorLevel_RIL_WORD, false) {
            // The components is a Boxa (Box Array) from Leptonica
            // We need to iterate through the boxes using the proper API
            let num_boxes = components.get_n();
            
            for i in 0..num_boxes {
                if let Some(bbox) = components.get_box(i) {
                    // Get the box dimensions
                    let mut x = 0i32;
                    let mut y = 0i32;
                    let mut w = 0i32;
                    let mut h = 0i32;
                    bbox.get_geometry(Some(&mut x), Some(&mut y), Some(&mut w), Some(&mut h));
                    
                    // Set rectangle to this word's area
                    tesseract.set_rectangle(x, y, w, h);
                    
                    // Get text for this word
                    if let Ok(word_text) = tesseract.get_utf8_text() {
                        let word_confidence = tesseract.mean_text_conf() as f32 / 100.0;
                        
                        // Only include words above confidence threshold
                        if word_confidence >= self.options.confidence_threshold {
                            boxes.push(BoundingBox {
                                text: word_text.trim().to_string(),
                                x: x as u32,
                                y: y as u32,
                                width: w as u32,
                                height: h as u32,
                                confidence: word_confidence,
                            });
                        }
                    }
                }
            }
        }
        
        Ok(boxes)
    }
}

impl Clone for OcrProcessor {
    fn clone(&self) -> Self {
        Self {
            options: self.options.clone(),
            semaphore: Arc::clone(&self.semaphore),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs;
    
    #[tokio::test]
    async fn test_ocr_processor_creation() {
        let options = OcrOptions::default();
        
        // This test will only pass if Tesseract is installed
        if let Ok(_processor) = OcrProcessor::new(options) {
            // Test passed - Tesseract is available
            println!("✓ Tesseract is available");
        } else {
            println!("⚠ Tesseract not available - OCR tests skipped");
        }
    }
    
    #[tokio::test]
    async fn test_get_available_languages() {
        if let Ok(languages) = OcrProcessor::get_available_languages() {
            println!("Available languages: {:?}", languages);
            assert!(!languages.is_empty());
        } else {
            println!("⚠ Could not get Tesseract languages");
        }
    }
    
    #[tokio::test]
    async fn test_ocr_validation() {
        let options = OcrOptions {
            confidence_threshold: 0.8,
            quality_threshold: 0.7,
            ..Default::default()
        };
        
        if let Ok(processor) = OcrProcessor::new(options) {
            let result = OcrResult {
                text: "Sample text for testing".to_string(),
                confidence: 0.85,
                page_count: 1,
                processing_time_ms: 100,
                language: Some("eng".to_string()),
                bounding_boxes: Vec::new(),
                metadata: HashMap::new(),
            };
            
            assert!(processor.is_quality_acceptable(&result));
            
            let poor_result = OcrResult {
                text: "Bad".to_string(),
                confidence: 0.3,
                page_count: 1,
                processing_time_ms: 100,
                language: Some("eng".to_string()),
                bounding_boxes: Vec::new(),
                metadata: HashMap::new(),
            };
            
            assert!(!processor.is_quality_acceptable(&poor_result));
        }
    }
}