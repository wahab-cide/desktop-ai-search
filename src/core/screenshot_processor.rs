use crate::error::{AppError, IndexingError, Result};
use crate::core::ocr_processor::OcrProcessor;
use crate::core::clip_processor::ClipProcessor;
use image::{DynamicImage, ImageFormat, GenericImageView};
use std::path::Path;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenshotMetadata {
    pub file_path: String,
    pub resolution: (u32, u32),
    pub aspect_ratio: f32,
    pub created_at: DateTime<Utc>,
    pub file_size: u64,
    pub format: String,
    pub is_screenshot: bool,
    pub confidence: f32,
    pub perceptual_hash: String,
    pub has_ui_elements: bool,
    pub has_text_content: bool,
    pub estimated_text_density: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenshotAnalysis {
    pub metadata: ScreenshotMetadata,
    pub extracted_text: Option<String>,
    pub text_confidence: f32,
    pub layout_analysis: LayoutAnalysis,
    pub visual_features: VisualFeatures,
    pub classification: ScreenshotType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutAnalysis {
    pub regions: Vec<LayoutRegion>,
    pub reading_order: Vec<usize>, // Indices into regions array
    pub has_structured_layout: bool,
    pub layout_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutRegion {
    pub region_type: RegionType,
    pub bbox: BoundingBox,
    pub text_content: Option<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegionType {
    Text,
    Image,
    UI,
    Menu,
    Button,
    Icon,
    Chart,
    Diagram,
    Table,
    Form,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualFeatures {
    pub dominant_colors: Vec<(u8, u8, u8)>, // RGB color palette
    pub average_brightness: f32,
    pub contrast_level: f32,
    pub edge_density: f32,
    pub color_variance: f32,
    pub has_borders: bool,
    pub ui_chrome_detected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScreenshotType {
    Desktop,
    Application,
    WebPage,
    Document,
    Diagram,
    Chart,
    Code,
    Chat,
    Unknown,
}

pub struct ScreenshotProcessor {
    ocr_processor: OcrProcessor,
    clip_processor: ClipProcessor,
    perceptual_hash_cache: HashMap<String, String>,
    screenshot_patterns: ScreenshotPatterns,
}

#[derive(Debug)]
struct ScreenshotPatterns {
    common_resolutions: Vec<(u32, u32)>,
    ui_indicators: Vec<String>,
    border_patterns: Vec<BorderPattern>,
}

#[derive(Debug)]
struct BorderPattern {
    color_range: ((u8, u8, u8), (u8, u8, u8)), // Min and max RGB
    thickness_range: (u32, u32),
    position: BorderPosition,
}

#[derive(Debug)]
enum BorderPosition {
    Top,
    Bottom,
    Left,
    Right,
    All,
}

impl ScreenshotProcessor {
    pub fn new() -> Result<Self> {
        let ocr_processor = OcrProcessor::new(Default::default())?;
        let clip_processor = ClipProcessor::new()?;
        
        Ok(Self {
            ocr_processor,
            clip_processor,
            perceptual_hash_cache: HashMap::new(),
            screenshot_patterns: ScreenshotPatterns::new(),
        })
    }

    /// Main entry point for processing screenshots and visual content
    pub async fn process_screenshot<P: AsRef<Path>>(&mut self, image_path: P) -> Result<ScreenshotAnalysis> {
        let path = image_path.as_ref();
        println!("Processing screenshot: {}", path.display());

        // Load and validate image
        let image = self.load_image(path).await?;
        
        // Extract basic metadata
        let metadata = self.extract_metadata(path, &image).await?;
        
        // Skip processing if not a screenshot with high confidence
        if !metadata.is_screenshot && metadata.confidence < 0.3 {
            return Ok(ScreenshotAnalysis {
                metadata,
                extracted_text: None,
                text_confidence: 0.0,
                layout_analysis: LayoutAnalysis::empty(),
                visual_features: self.analyze_visual_features(&image)?,
                classification: ScreenshotType::Unknown,
            });
        }

        // Perform dual-pass OCR strategy
        let (extracted_text, text_confidence) = self.dual_pass_ocr(&image, &metadata).await?;
        
        // Analyze layout and structure
        let layout_analysis = self.analyze_layout(&image, &extracted_text).await?;
        
        // Extract visual features
        let visual_features = self.analyze_visual_features(&image)?;
        
        // Classify screenshot type
        let classification = self.classify_screenshot(&metadata, &layout_analysis, &visual_features);

        Ok(ScreenshotAnalysis {
            metadata,
            extracted_text,
            text_confidence,
            layout_analysis,
            visual_features,
            classification,
        })
    }

    /// Dual-pass OCR strategy: lightweight detection first, then full OCR
    async fn dual_pass_ocr(&self, image: &DynamicImage, metadata: &ScreenshotMetadata) -> Result<(Option<String>, f32)> {
        // Pass 1: Quick text detection using Tesseract PSM 7 on downscaled image
        let quick_detection_result = self.quick_text_detection(image).await?;
        
        if quick_detection_result.ascii_hit_rate < 0.05 {
            // Low text density detected, skip full OCR
            return Ok((None, 0.0));
        }

        // Pass 2: Full layout-aware OCR using PaddleOCR
        println!("Text detected, performing full OCR...");
        let ocr_result = self.ocr_processor.process_image_bytes(&self.image_to_bytes(image)?, None).await?;
        
        let confidence = ocr_result.confidence;
        let text = if confidence > 0.7 { 
            Some(ocr_result.text) 
        } else { 
            None 
        };

        Ok((text, confidence))
    }

    /// Quick text detection using downscaled image and simple heuristics
    async fn quick_text_detection(&self, image: &DynamicImage) -> Result<QuickDetectionResult> {
        // Downscale image to 400px width for quick processing
        let scale_factor = 400.0 / image.width() as f32;
        let new_height = (image.height() as f32 * scale_factor) as u32;
        let downscaled = image.resize(400, new_height, image::imageops::FilterType::Lanczos3);
        
        // Convert to grayscale and apply basic text detection heuristics
        let gray_image = downscaled.to_luma8();
        
        // Simple edge density calculation as text indicator
        let edge_count = self.calculate_edge_density(&gray_image);
        let total_pixels = gray_image.width() * gray_image.height();
        let edge_density = edge_count as f32 / total_pixels as f32;
        
        // Estimate ASCII hit rate based on edge patterns
        let ascii_hit_rate = if edge_density > 0.1 { 
            (edge_density * 2.0).min(1.0) 
        } else { 
            0.0 
        };

        Ok(QuickDetectionResult {
            ascii_hit_rate,
            edge_density,
            likely_has_text: ascii_hit_rate > 0.05,
        })
    }

    /// Calculate edge density for text detection
    fn calculate_edge_density(&self, gray_image: &image::GrayImage) -> u32 {
        let (width, height) = gray_image.dimensions();
        let mut edge_count = 0;

        // Simple Sobel edge detection
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let _pixel = gray_image.get_pixel(x, y)[0] as i32;
                
                // Calculate gradients
                let gx = (gray_image.get_pixel(x + 1, y)[0] as i32 - gray_image.get_pixel(x - 1, y)[0] as i32);
                let gy = (gray_image.get_pixel(x, y + 1)[0] as i32 - gray_image.get_pixel(x, y - 1)[0] as i32);
                
                let magnitude = ((gx * gx + gy * gy) as f32).sqrt() as u32;
                
                if magnitude > 50 { // Threshold for edge detection
                    edge_count += 1;
                }
            }
        }

        edge_count
    }

    /// Load and validate image file
    async fn load_image<P: AsRef<Path>>(&self, path: P) -> Result<DynamicImage> {
        let image = image::open(path.as_ref())
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to load image: {}", e)
            )))?;
        
        // Validate image dimensions (prevent processing of tiny or huge images)
        let (width, height) = image.dimensions();
        if width < 50 || height < 50 {
            return Err(AppError::Indexing(IndexingError::Processing(
                "Image too small for screenshot processing".to_string()
            )));
        }
        
        if width > 8192 || height > 8192 {
            return Err(AppError::Indexing(IndexingError::Processing(
                "Image too large for screenshot processing".to_string()
            )));
        }

        Ok(image)
    }

    /// Extract metadata and detect if image is likely a screenshot
    async fn extract_metadata<P: AsRef<Path>>(&mut self, path: P, image: &DynamicImage) -> Result<ScreenshotMetadata> {
        let path = path.as_ref();
        let (width, height) = image.dimensions();
        let aspect_ratio = width as f32 / height as f32;
        
        // Get file metadata
        let file_metadata = std::fs::metadata(path)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to read file metadata: {}", e)
            )))?;
        
        let file_size = file_metadata.len();
        let created_at = file_metadata.created()
            .map(|t| DateTime::<Utc>::from(t))
            .unwrap_or_else(|_| Utc::now());

        // Detect image format
        let format = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("unknown")
            .to_uppercase();

        // Calculate perceptual hash for deduplication
        let perceptual_hash = self.calculate_perceptual_hash(image)?;
        
        // Check for duplicate using perceptual hash
        if let Some(existing_path) = self.perceptual_hash_cache.get(&perceptual_hash) {
            println!("Duplicate screenshot detected: {} (original: {})", path.display(), existing_path);
        } else {
            self.perceptual_hash_cache.insert(perceptual_hash.clone(), path.to_string_lossy().to_string());
        }

        // Detect if this looks like a screenshot
        let (is_screenshot, confidence) = self.detect_screenshot_heuristics(image, &aspect_ratio);
        
        // Quick UI element detection
        let has_ui_elements = self.detect_ui_elements(image);
        
        // Estimate text density
        let estimated_text_density = self.estimate_text_density(image).await?;
        let has_text_content = estimated_text_density > 0.1;

        Ok(ScreenshotMetadata {
            file_path: path.to_string_lossy().to_string(),
            resolution: (width, height),
            aspect_ratio,
            created_at,
            file_size,
            format,
            is_screenshot,
            confidence,
            perceptual_hash,
            has_ui_elements,
            has_text_content,
            estimated_text_density,
        })
    }

    /// Detect if image is likely a screenshot using heuristics
    fn detect_screenshot_heuristics(&self, image: &DynamicImage, aspect_ratio: &f32) -> (bool, f32) {
        let (width, height) = image.dimensions();
        let mut confidence_score: f32 = 0.0;
        let mut is_screenshot = false;

        // Check common screenshot resolutions
        if self.screenshot_patterns.common_resolutions.contains(&(width, height)) {
            confidence_score += 0.4;
            is_screenshot = true;
        }

        // Check aspect ratios common for monitors
        let common_ratios = [16.0/9.0, 16.0/10.0, 4.0/3.0, 21.0/9.0, 3.0/2.0];
        for ratio in &common_ratios {
            if (aspect_ratio - ratio).abs() < 0.05 {
                confidence_score += 0.2;
                if !is_screenshot && confidence_score > 0.2 {
                    is_screenshot = true;
                }
                break;
            }
        }

        // Detect solid-color borders (common in screenshots)
        if self.has_uniform_borders(image) {
            confidence_score += 0.3;
            is_screenshot = true;
        }

        // Check for UI chrome patterns (title bars, status bars)
        if self.detect_ui_chrome(image) {
            confidence_score += 0.4;
            is_screenshot = true;
        }

        (is_screenshot, confidence_score.min(1.0))
    }

    /// Calculate perceptual hash for duplicate detection
    fn calculate_perceptual_hash(&self, image: &DynamicImage) -> Result<String> {
        // Resize to 8x8 for pHash calculation
        let small_image = image.resize_exact(8, 8, image::imageops::FilterType::Lanczos3);
        let gray_image = small_image.to_luma8();
        
        // Calculate DCT-based perceptual hash
        let mut hash_bits = Vec::new();
        let pixels: Vec<f32> = gray_image.pixels().map(|p| p[0] as f32).collect();
        
        // Simple average-based hash (could be enhanced with DCT)
        let average: f32 = pixels.iter().sum::<f32>() / pixels.len() as f32;
        
        for pixel_value in pixels {
            hash_bits.push(if pixel_value >= average { '1' } else { '0' });
        }
        
        Ok(hash_bits.into_iter().collect())
    }

    /// Detect UI elements like buttons, menus, icons
    fn detect_ui_elements(&self, image: &DynamicImage) -> bool {
        // Simple heuristic: look for rectangular regions with uniform colors
        // In a real implementation, this would use more sophisticated computer vision
        
        let (width, height) = image.dimensions();
        
        // Check top and bottom edges for UI chrome
        let top_region_uniform = self.check_region_uniformity(image, 0, 0, width, height / 10);
        let bottom_region_uniform = self.check_region_uniformity(image, 0, height * 9 / 10, width, height / 10);
        
        top_region_uniform || bottom_region_uniform
    }

    /// Check if a region has uniform color (potential UI element)
    fn check_region_uniformity(&self, image: &DynamicImage, x: u32, y: u32, w: u32, h: u32) -> bool {
        if w == 0 || h == 0 {
            return false;
        }

        let mut color_counts = HashMap::new();
        let total_pixels = w * h;
        
        for py in y..std::cmp::min(y + h, image.height()) {
            for px in x..std::cmp::min(x + w, image.width()) {
                let pixel = image.get_pixel(px, py);
                let color_key = (pixel[0], pixel[1], pixel[2]); // RGB
                *color_counts.entry(color_key).or_insert(0) += 1;
            }
        }
        
        // Check if dominant color covers >70% of the region
        if let Some(max_count) = color_counts.values().max() {
            *max_count as f32 / total_pixels as f32 > 0.7
        } else {
            false
        }
    }

    /// Detect uniform borders around the image
    fn has_uniform_borders(&self, image: &DynamicImage) -> bool {
        let (width, height) = image.dimensions();
        let border_thickness = std::cmp::min(10, std::cmp::min(width, height) / 20);
        
        // Check if borders are uniform
        let top_uniform = self.check_region_uniformity(image, 0, 0, width, border_thickness);
        let bottom_uniform = self.check_region_uniformity(image, 0, height - border_thickness, width, border_thickness);
        let left_uniform = self.check_region_uniformity(image, 0, 0, border_thickness, height);
        let right_uniform = self.check_region_uniformity(image, width - border_thickness, 0, border_thickness, height);
        
        [top_uniform, bottom_uniform, left_uniform, right_uniform].iter().filter(|&&x| x).count() >= 2
    }

    /// Detect UI chrome like title bars and toolbars
    fn detect_ui_chrome(&self, image: &DynamicImage) -> bool {
        let (width, height) = image.dimensions();
        
        // Look for horizontal regions that might be title bars or toolbars
        let title_bar_height = height / 20; // Assume title bar is about 5% of height
        
        // Check top region for title bar pattern
        if title_bar_height > 0 {
            let top_uniform = self.check_region_uniformity(image, 0, 0, width, title_bar_height);
            if top_uniform {
                return true;
            }
        }
        
        // Check for menu bar patterns (typically darker regions with buttons)
        let menu_bar_start = title_bar_height;
        let menu_bar_height = height / 25;
        
        if menu_bar_height > 0 && menu_bar_start + menu_bar_height < height {
            let menu_uniform = self.check_region_uniformity(image, 0, menu_bar_start, width, menu_bar_height);
            if menu_uniform {
                return true;
            }
        }
        
        false
    }

    /// Estimate text density in the image
    async fn estimate_text_density(&self, image: &DynamicImage) -> Result<f32> {
        // Quick estimation based on edge density patterns
        let gray_image = image.to_luma8();
        let edge_density = self.calculate_edge_density(&gray_image);
        let total_pixels = gray_image.width() * gray_image.height();
        
        // Normalize edge density to approximate text density
        let raw_density = edge_density as f32 / total_pixels as f32;
        
        // Apply scaling factor based on empirical observation
        let estimated_density = (raw_density * 5.0).min(1.0);
        
        Ok(estimated_density)
    }

    /// Analyze layout and structure of the screenshot
    async fn analyze_layout(&self, image: &DynamicImage, extracted_text: &Option<String>) -> Result<LayoutAnalysis> {
        // Simplified layout analysis - in production this would use more sophisticated algorithms
        let mut regions = Vec::new();
        let (width, height) = image.dimensions();
        
        // Create basic grid-based regions for analysis
        let grid_size = 4;
        let cell_width = width / grid_size;
        let cell_height = height / grid_size;
        
        for row in 0..grid_size {
            for col in 0..grid_size {
                let x = col * cell_width;
                let y = row * cell_height;
                
                let region_type = self.classify_region_type(image, x, y, cell_width, cell_height);
                let confidence = self.calculate_region_confidence(&region_type);
                
                regions.push(LayoutRegion {
                    region_type,
                    bbox: BoundingBox {
                        x,
                        y,
                        width: cell_width,
                        height: cell_height,
                    },
                    text_content: None, // Would be populated by more detailed OCR
                    confidence,
                });
            }
        }
        
        // Create simple reading order (left-to-right, top-to-bottom)
        let reading_order: Vec<usize> = (0..regions.len()).collect();
        
        let has_structured_layout = regions.iter()
            .any(|r| matches!(r.region_type, RegionType::UI | RegionType::Menu | RegionType::Button));
        
        let layout_confidence = if has_structured_layout { 0.7 } else { 0.3 };

        Ok(LayoutAnalysis {
            regions,
            reading_order,
            has_structured_layout,
            layout_confidence,
        })
    }

    /// Classify the type of content in a specific region
    fn classify_region_type(&self, image: &DynamicImage, x: u32, y: u32, w: u32, h: u32) -> RegionType {
        // Check if region is uniform (likely UI)
        if self.check_region_uniformity(image, x, y, w, h) {
            return RegionType::UI;
        }
        
        // Calculate color variance to detect different content types
        let color_variance = self.calculate_color_variance(image, x, y, w, h);
        
        if color_variance < 0.1 {
            RegionType::UI
        } else if color_variance > 0.8 {
            RegionType::Image
        } else {
            RegionType::Text
        }
    }

    /// Calculate color variance in a region
    fn calculate_color_variance(&self, image: &DynamicImage, x: u32, y: u32, w: u32, h: u32) -> f32 {
        let mut color_values = Vec::new();
        
        for py in y..std::cmp::min(y + h, image.height()) {
            for px in x..std::cmp::min(x + w, image.width()) {
                let pixel = image.get_pixel(px, py);
                let luminance = 0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
                color_values.push(luminance);
            }
        }
        
        if color_values.is_empty() {
            return 0.0;
        }
        
        let mean = color_values.iter().sum::<f32>() / color_values.len() as f32;
        let variance = color_values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / color_values.len() as f32;
        
        variance / 255.0 // Normalize to 0-1 range
    }

    /// Calculate confidence score for region classification
    fn calculate_region_confidence(&self, region_type: &RegionType) -> f32 {
        match region_type {
            RegionType::UI => 0.8,
            RegionType::Text => 0.7,
            RegionType::Image => 0.6,
            _ => 0.5,
        }
    }

    /// Analyze visual features of the image
    fn analyze_visual_features(&self, image: &DynamicImage) -> Result<VisualFeatures> {
        let (width, height) = image.dimensions();
        
        // Calculate dominant colors (simplified - just sample some pixels)
        let mut color_counts = HashMap::new();
        let sample_step = std::cmp::max(1, std::cmp::max(width, height) / 100);
        
        for y in (0..height).step_by(sample_step as usize) {
            for x in (0..width).step_by(sample_step as usize) {
                let pixel = image.get_pixel(x, y);
                let color = (pixel[0], pixel[1], pixel[2]);
                *color_counts.entry(color).or_insert(0) += 1;
            }
        }
        
        let mut dominant_colors: Vec<_> = color_counts.into_iter()
            .collect::<Vec<_>>();
        dominant_colors.sort_by(|a, b| b.1.cmp(&a.1));
        let dominant_colors: Vec<_> = dominant_colors.into_iter()
            .take(5)
            .map(|(color, _)| color)
            .collect();

        // Calculate average brightness
        let gray_image = image.to_luma8();
        let total_brightness: u64 = gray_image.pixels()
            .map(|p| p[0] as u64)
            .sum();
        let average_brightness = total_brightness as f32 / (width * height) as f32 / 255.0;

        // Calculate edge density for contrast estimation
        let edge_count = self.calculate_edge_density(&gray_image);
        let edge_density = edge_count as f32 / (width * height) as f32;

        // Calculate color variance for contrast level
        let color_variance = self.calculate_color_variance(image, 0, 0, width, height);
        let contrast_level = color_variance;

        // Detect borders and UI chrome
        let has_borders = self.has_uniform_borders(image);
        let ui_chrome_detected = self.detect_ui_chrome(image);

        Ok(VisualFeatures {
            dominant_colors,
            average_brightness,
            contrast_level,
            edge_density,
            color_variance,
            has_borders,
            ui_chrome_detected,
        })
    }

    /// Classify the type of screenshot based on analysis
    fn classify_screenshot(&self, metadata: &ScreenshotMetadata, layout: &LayoutAnalysis, features: &VisualFeatures) -> ScreenshotType {
        // Use heuristics to classify screenshot type
        if features.ui_chrome_detected {
            return ScreenshotType::Application;
        }
        
        if layout.has_structured_layout {
            if features.average_brightness > 0.8 {
                ScreenshotType::Document
            } else {
                ScreenshotType::WebPage
            }
        } else if features.edge_density > 0.3 {
            ScreenshotType::Diagram
        } else if metadata.aspect_ratio > 2.0 {
            ScreenshotType::Chart
        } else {
            ScreenshotType::Desktop
        }
    }

    /// Convert image to bytes for OCR processing
    fn image_to_bytes(&self, image: &DynamicImage) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        image.write_to(&mut std::io::Cursor::new(&mut bytes), ImageFormat::Png)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to convert image to bytes: {}", e)
            )))?;
        Ok(bytes)
    }

    /// Search for visually similar images using CLIP embeddings
    pub async fn find_similar_screenshots(&mut self, query_image_path: &str, image_database: &[(String, Vec<f32>)], top_k: usize) -> Result<Vec<VisualSimilarityResult>> {
        // Generate embedding for query image
        let query_embeddings = self.clip_processor.generate_image_embeddings(&[query_image_path.to_string()]).await?;
        let query_embedding = &query_embeddings[0];
        
        // Calculate similarities
        let mut similarities: Vec<VisualSimilarityResult> = image_database
            .iter()
            .map(|(image_path, image_embedding)| {
                let similarity = self.clip_processor.cosine_similarity(query_embedding, image_embedding);
                VisualSimilarityResult {
                    image_path: image_path.clone(),
                    similarity_score: similarity,
                    match_type: if similarity > 0.9 { 
                        MatchType::Duplicate 
                    } else if similarity > 0.7 { 
                        MatchType::VerySimilar 
                    } else if similarity > 0.5 { 
                        MatchType::Similar 
                    } else { 
                        MatchType::Related 
                    },
                }
            })
            .collect();

        // Sort by similarity and take top k
        similarities.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        
        Ok(similarities)
    }

    /// Clear the perceptual hash cache
    pub fn clear_cache(&mut self) {
        self.perceptual_hash_cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        (self.perceptual_hash_cache.len(), 1000) // Assuming max cache size of 1000
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualSimilarityResult {
    pub image_path: String,
    pub similarity_score: f32,
    pub match_type: MatchType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchType {
    Duplicate,
    VerySimilar,
    Similar,
    Related,
}

#[derive(Debug)]
struct QuickDetectionResult {
    ascii_hit_rate: f32,
    edge_density: f32,
    likely_has_text: bool,
}

impl ScreenshotPatterns {
    fn new() -> Self {
        Self {
            common_resolutions: vec![
                // Common monitor resolutions
                (1920, 1080), (2560, 1440), (1366, 768), (1440, 900),
                (1680, 1050), (1280, 720), (1600, 900), (2560, 1600),
                (3840, 2160), (2880, 1800), (1280, 800), (1024, 768),
                // MacBook resolutions
                (2880, 1864), (2560, 1664), (2560, 1600), (1440, 900),
                // iPhone resolutions (for mobile screenshots)
                (1284, 2778), (1170, 2532), (828, 1792), (750, 1334),
            ],
            ui_indicators: vec![
                "title bar".to_string(),
                "menu bar".to_string(),
                "status bar".to_string(),
                "toolbar".to_string(),
            ],
            border_patterns: vec![
                BorderPattern {
                    color_range: ((200, 200, 200), (255, 255, 255)), // Light gray to white
                    thickness_range: (1, 5),
                    position: BorderPosition::All,
                },
                BorderPattern {
                    color_range: ((0, 0, 0), (50, 50, 50)), // Black to dark gray
                    thickness_range: (1, 3),
                    position: BorderPosition::All,
                },
            ],
        }
    }
}

impl LayoutAnalysis {
    fn empty() -> Self {
        Self {
            regions: Vec::new(),
            reading_order: Vec::new(),
            has_structured_layout: false,
            layout_confidence: 0.0,
        }
    }
}

impl Default for ScreenshotProcessor {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

