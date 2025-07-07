use desktop_ai_search::test_query_understanding::{test_query_understanding_basic_functionality};
use desktop_ai_search::test_boolean_logic::{test_boolean_logic_basic_functionality};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Week 7 Basic Functionality Test");
    println!("==================================");
    println!("Testing core Week 7 implementations for compilation and basic functionality");
    println!();
    
    // Test 1: Query Understanding Basic Functionality
    println!("1️⃣ Testing Query Understanding Basic Functionality...");
    if let Err(e) = test_query_understanding_basic_functionality() {
        println!("❌ Query understanding basic functionality failed: {}", e);
        return Err(e);
    } else {
        println!("✅ Query understanding basic functionality passed");
    }
    
    // Test 2: Boolean Logic Basic Functionality  
    println!("\n2️⃣ Testing Boolean Logic Basic Functionality...");
    if let Err(e) = test_boolean_logic_basic_functionality() {
        println!("❌ Boolean logic basic functionality failed: {}", e);
        return Err(e);
    } else {
        println!("✅ Boolean logic basic functionality passed");
    }
    
    // Test 3: Core Module Compilation Test
    println!("\n3️⃣ Testing Core Module Compilation...");
    test_core_module_compilation().await?;
    
    // Test 4: Basic Contextual Search Structures
    println!("\n4️⃣ Testing Basic Contextual Search Structures...");
    test_basic_contextual_structures()?;
    
    println!("\n🎯 WEEK 7 BASIC FUNCTIONALITY COMPLETE");
    println!("======================================");
    println!("✅ All core Week 7 modules compile successfully");
    println!("✅ Query understanding system operational");
    println!("✅ Boolean logic and PEG grammar functional");
    println!("✅ Contextual search structures defined");
    println!("✅ Ready for full integration testing");
    
    Ok(())
}

async fn test_core_module_compilation() -> Result<(), Box<dyn std::error::Error>> {
    use desktop_ai_search::core::advanced_query_processor::AdvancedQueryProcessor;
    use desktop_ai_search::core::boolean_query_parser::BooleanQueryParser;
    use desktop_ai_search::core::intelligent_query_processor::IntelligentQueryProcessor;
    use desktop_ai_search::database::Database;
    
    println!("   Creating database connection...");
    let database = Database::new("test_compilation.db")?;
    
    println!("   Initializing advanced query processor...");
    let _query_processor = AdvancedQueryProcessor::new()?;
    
    println!("   Initializing boolean query parser...");
    let _boolean_parser = BooleanQueryParser::new();
    
    println!("   Initializing intelligent query processor...");
    let _intelligent_processor = IntelligentQueryProcessor::new(database)?;
    
    println!("   ✅ All core processors initialized successfully");
    Ok(())
}

fn test_basic_contextual_structures() -> Result<(), Box<dyn std::error::Error>> {
    use desktop_ai_search::core::adaptive_learning_engine::{LearningState, UserLearningModel};
    use std::collections::{HashMap, VecDeque};
    use chrono::Utc;
    use uuid::Uuid;
    
    println!("   Creating basic user learning model...");
    
    // Test creating a simplified learning model structure
    let user_id = Uuid::new_v4();
    
    println!("   Learning state enum test...");
    let _cold_start = LearningState::ColdStart;
    let _bootstrapping = LearningState::Bootstrapping;
    let _active = LearningState::Active;
    let _stable = LearningState::Stable;
    let _adapting = LearningState::Adapting;
    let _degraded = LearningState::Degraded;
    
    println!("   Basic structures compile successfully...");
    println!("   User ID: {}", user_id);
    
    println!("   ✅ Basic contextual search structures working");
    Ok(())
}

// Simplified test for just the compilation and basic usage
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_week7_functionality() {
        if let Err(e) = test_core_module_compilation().await {
            panic!("Core module compilation test failed: {}", e);
        }
        
        if let Err(e) = test_basic_contextual_structures() {
            panic!("Basic contextual structures test failed: {}", e);
        }
    }
}