use crate::error::Result;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Boolean query parser using PEG grammar for complex search expressions
/// Supports: AND, OR, NOT, parentheses, field-specific search, phrases, wildcards
#[derive(Debug, Clone)]
pub struct BooleanQueryParser {
    field_aliases: HashMap<String, String>,
    operator_aliases: HashMap<String, BooleanOperator>,
    default_field: String,
    case_sensitive: bool,
}

/// Parsed query tree structure representing complex boolean expressions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QueryNode {
    /// Terminal node: actual search term
    Term(TermQuery),
    /// Binary operation: left AND/OR right
    Binary {
        left: Box<QueryNode>,
        operator: BooleanOperator,
        right: Box<QueryNode>,
    },
    /// Unary operation: NOT expression
    Not(Box<QueryNode>),
    /// Grouped expression: (...)
    Group(Box<QueryNode>),
    /// Empty query
    Empty,
}

/// Terminal search term with field specification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TermQuery {
    pub field: Option<String>,
    pub value: String,
    pub query_type: TermType,
    pub boost: Option<f32>,
    pub fuzzy: Option<FuzzyConfig>,
    pub proximity: Option<u32>, // For phrase queries
}

/// Types of terminal queries
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TermType {
    /// Exact word match
    Word,
    /// Exact phrase match (quoted)
    Phrase,
    /// Wildcard pattern (* and ?)
    Wildcard,
    /// Regular expression
    Regex,
    /// Range query (numbers, dates)
    Range { start: String, end: String, inclusive: bool },
    /// Existence check (field exists)
    Exists,
}

/// Boolean operators with precedence
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum BooleanOperator {
    And,    // Precedence: 2
    Or,     // Precedence: 1  
    Not,    // Precedence: 3 (highest)
}

/// Fuzzy search configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FuzzyConfig {
    pub distance: u32,      // Edit distance
    pub prefix_length: u32, // Prefix that must match exactly
}

/// Query parsing result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedQuery {
    pub tree: QueryNode,
    pub original_query: String,
    pub normalized_query: String,
    pub complexity_score: f32,
    pub field_usage: HashMap<String, u32>,
    pub operator_usage: HashMap<BooleanOperator, u32>,
    pub estimated_selectivity: f32,
    pub parsing_time_ms: f64,
}

/// Query optimization hints
#[derive(Debug, Clone)]
pub struct QueryOptimization {
    pub rewritten_tree: QueryNode,
    pub applied_optimizations: Vec<OptimizationType>,
    pub estimated_performance_gain: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    /// Reorder terms for better selectivity
    TermReordering,
    /// Convert OR to AND where possible
    OperatorRewriting,
    /// Remove redundant NOT NOT
    DoubleNegationElimination,
    /// Factor out common terms
    CommonTermFactoring,
    /// Push NOT operations down
    NotPushdown,
    /// Remove always-true/false conditions
    ConstantFolding,
}

/// Token types for lexical analysis
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Word(String),
    Phrase(String),
    FieldSpecifier { field: String, value: String },
    Operator(BooleanOperator),
    LeftParen,
    RightParen,
    Wildcard(String),
    Range { start: String, end: String, inclusive: bool },
    Boost(f32),
    Fuzzy { term: String, distance: u32 },
    Eof,
}

/// Lexer for tokenizing query strings
#[derive(Debug)]
struct QueryLexer {
    input: Vec<char>,
    position: usize,
    current_char: Option<char>,
}

/// Parser state for recursive descent parsing
#[derive(Debug)]
struct QueryParser {
    lexer: QueryLexer,
    current_token: Token,
    field_aliases: HashMap<String, String>,
    operator_aliases: HashMap<String, BooleanOperator>,
}

impl BooleanQueryParser {
    pub fn new() -> Self {
        let mut field_aliases = HashMap::new();
        field_aliases.insert("author".to_string(), "creator".to_string());
        field_aliases.insert("from".to_string(), "creator".to_string());
        field_aliases.insert("by".to_string(), "creator".to_string());
        field_aliases.insert("type".to_string(), "file_type".to_string());
        field_aliases.insert("ext".to_string(), "file_type".to_string());
        field_aliases.insert("extension".to_string(), "file_type".to_string());
        field_aliases.insert("modified".to_string(), "modified_date".to_string());
        field_aliases.insert("created".to_string(), "created_date".to_string());
        field_aliases.insert("date".to_string(), "modified_date".to_string());
        field_aliases.insert("size".to_string(), "file_size".to_string());
        field_aliases.insert("title".to_string(), "document_title".to_string());
        field_aliases.insert("content".to_string(), "text_content".to_string());
        field_aliases.insert("text".to_string(), "text_content".to_string());
        
        let mut operator_aliases = HashMap::new();
        operator_aliases.insert("AND".to_string(), BooleanOperator::And);
        operator_aliases.insert("and".to_string(), BooleanOperator::And);
        operator_aliases.insert("&".to_string(), BooleanOperator::And);
        operator_aliases.insert("&&".to_string(), BooleanOperator::And);
        operator_aliases.insert("OR".to_string(), BooleanOperator::Or);
        operator_aliases.insert("or".to_string(), BooleanOperator::Or);
        operator_aliases.insert("|".to_string(), BooleanOperator::Or);
        operator_aliases.insert("||".to_string(), BooleanOperator::Or);
        operator_aliases.insert("NOT".to_string(), BooleanOperator::Not);
        operator_aliases.insert("not".to_string(), BooleanOperator::Not);
        operator_aliases.insert("!".to_string(), BooleanOperator::Not);
        operator_aliases.insert("-".to_string(), BooleanOperator::Not);

        Self {
            field_aliases,
            operator_aliases,
            default_field: "text_content".to_string(),
            case_sensitive: false,
        }
    }

    /// Parse a boolean query string into a structured query tree
    pub fn parse(&self, query: &str) -> Result<ParsedQuery> {
        let start_time = std::time::Instant::now();
        let original_query = query.to_string();
        
        println!("Parsing boolean query: '{}'", query);
        
        // Normalize the query
        let normalized_query = self.normalize_query(query);
        
        // Tokenize the input
        let mut lexer = QueryLexer::new(&normalized_query);
        let mut parser = QueryParser::new(lexer, &self.field_aliases, &self.operator_aliases);
        
        // Parse into AST
        let tree = parser.parse_expression()?;
        
        // Calculate metadata
        let complexity_score = self.calculate_complexity(&tree);
        let field_usage = self.analyze_field_usage(&tree);
        let operator_usage = self.analyze_operator_usage(&tree);
        let estimated_selectivity = self.estimate_selectivity(&tree);
        
        let parsing_time_ms = start_time.elapsed().as_millis() as f64;
        
        Ok(ParsedQuery {
            tree,
            original_query,
            normalized_query,
            complexity_score,
            field_usage,
            operator_usage,
            estimated_selectivity,
            parsing_time_ms,
        })
    }

    /// Optimize a parsed query for better performance
    pub fn optimize(&self, query: &ParsedQuery) -> QueryOptimization {
        let mut optimized_tree = query.tree.clone();
        let mut applied_optimizations = Vec::new();
        
        // Apply various optimizations
        optimized_tree = self.eliminate_double_negation(optimized_tree, &mut applied_optimizations);
        optimized_tree = self.constant_folding(optimized_tree, &mut applied_optimizations);
        optimized_tree = self.reorder_terms(optimized_tree, &mut applied_optimizations);
        optimized_tree = self.push_not_down(optimized_tree, &mut applied_optimizations);
        optimized_tree = self.factor_common_terms(optimized_tree, &mut applied_optimizations);
        
        let estimated_performance_gain = applied_optimizations.len() as f32 * 0.15; // Rough estimate
        
        QueryOptimization {
            rewritten_tree: optimized_tree,
            applied_optimizations,
            estimated_performance_gain,
        }
    }

    /// Convert query tree back to normalized string representation
    pub fn tree_to_string(&self, tree: &QueryNode) -> String {
        match tree {
            QueryNode::Empty => "".to_string(),
            QueryNode::Term(term) => self.term_to_string(term),
            QueryNode::Binary { left, operator, right } => {
                let left_str = self.tree_to_string(left);
                let right_str = self.tree_to_string(right);
                let op_str = match operator {
                    BooleanOperator::And => " AND ",
                    BooleanOperator::Or => " OR ",
                    BooleanOperator::Not => " NOT ", // This shouldn't happen in binary
                };
                format!("({} {} {})", left_str, op_str.trim(), right_str)
            },
            QueryNode::Not(expr) => {
                format!("NOT ({})", self.tree_to_string(expr))
            },
            QueryNode::Group(expr) => {
                format!("({})", self.tree_to_string(expr))
            },
        }
    }

    /// Normalize query string for consistent parsing
    fn normalize_query(&self, query: &str) -> String {
        let mut normalized = query.trim().to_string();
        
        // Replace smart quotes with regular quotes
        normalized = normalized.replace('"', "\"").replace('"', "\"");
        
        // Normalize whitespace
        normalized = regex::Regex::new(r"\s+").unwrap()
            .replace_all(&normalized, " ").to_string();
        
        // Handle implicit AND operations
        normalized = self.add_implicit_ands(&normalized);
        
        normalized
    }

    /// Add implicit AND operators where missing
    fn add_implicit_ands(&self, query: &str) -> String {
        // This is a simplified implementation
        // In practice, you'd want more sophisticated logic
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut result = Vec::new();
        
        for (i, word) in words.iter().enumerate() {
            result.push(word.to_string());
            
            // Add AND if next word isn't an operator and current isn't an operator
            if i < words.len() - 1 {
                let current_is_op = self.operator_aliases.contains_key(*word);
                let next_is_op = self.operator_aliases.contains_key(words[i + 1]);
                let current_is_paren = word == &"(" || word == &")";
                let next_is_paren = words[i + 1] == "(" || words[i + 1] == ")";
                
                if !current_is_op && !next_is_op && !current_is_paren && !next_is_paren {
                    result.push("AND".to_string());
                }
            }
        }
        
        result.join(" ")
    }

    /// Calculate query complexity score
    fn calculate_complexity(&self, tree: &QueryNode) -> f32 {
        match tree {
            QueryNode::Empty => 0.0,
            QueryNode::Term(_) => 1.0,
            QueryNode::Binary { left, right, .. } => {
                1.0 + self.calculate_complexity(left) + self.calculate_complexity(right)
            },
            QueryNode::Not(expr) => 1.5 + self.calculate_complexity(expr),
            QueryNode::Group(expr) => 0.5 + self.calculate_complexity(expr),
        }
    }

    /// Analyze field usage in the query
    fn analyze_field_usage(&self, tree: &QueryNode) -> HashMap<String, u32> {
        let mut usage = HashMap::new();
        self.collect_field_usage(tree, &mut usage);
        usage
    }

    fn collect_field_usage(&self, tree: &QueryNode, usage: &mut HashMap<String, u32>) {
        match tree {
            QueryNode::Term(term) => {
                let field = term.field.as_ref()
                    .unwrap_or(&self.default_field);
                *usage.entry(field.clone()).or_insert(0) += 1;
            },
            QueryNode::Binary { left, right, .. } => {
                self.collect_field_usage(left, usage);
                self.collect_field_usage(right, usage);
            },
            QueryNode::Not(expr) | QueryNode::Group(expr) => {
                self.collect_field_usage(expr, usage);
            },
            QueryNode::Empty => {},
        }
    }

    /// Analyze operator usage in the query
    fn analyze_operator_usage(&self, tree: &QueryNode) -> HashMap<BooleanOperator, u32> {
        let mut usage = HashMap::new();
        self.collect_operator_usage(tree, &mut usage);
        usage
    }

    fn collect_operator_usage(&self, tree: &QueryNode, usage: &mut HashMap<BooleanOperator, u32>) {
        match tree {
            QueryNode::Binary { left, operator, right } => {
                *usage.entry(operator.clone()).or_insert(0) += 1;
                self.collect_operator_usage(left, usage);
                self.collect_operator_usage(right, usage);
            },
            QueryNode::Not(expr) => {
                *usage.entry(BooleanOperator::Not).or_insert(0) += 1;
                self.collect_operator_usage(expr, usage);
            },
            QueryNode::Group(expr) => {
                self.collect_operator_usage(expr, usage);
            },
            _ => {},
        }
    }

    /// Estimate query selectivity (lower = more selective)
    fn estimate_selectivity(&self, tree: &QueryNode) -> f32 {
        match tree {
            QueryNode::Empty => 1.0,
            QueryNode::Term(term) => {
                // Estimate based on term type and field
                match term.query_type {
                    TermType::Phrase => 0.1,      // Phrases are very selective
                    TermType::Word => 0.3,        // Words are moderately selective
                    TermType::Wildcard => 0.7,    // Wildcards are less selective
                    TermType::Exists => 0.8,      // Existence checks are broad
                    TermType::Range { .. } => 0.5, // Ranges vary
                    TermType::Regex => 0.6,       // Regex varies
                }
            },
            QueryNode::Binary { left, operator, right } => {
                let left_sel = self.estimate_selectivity(left);
                let right_sel = self.estimate_selectivity(right);
                match operator {
                    BooleanOperator::And => left_sel * right_sel,  // More selective
                    BooleanOperator::Or => left_sel + right_sel - (left_sel * right_sel), // Less selective
                    BooleanOperator::Not => unreachable!(), // NOT is unary
                }
            },
            QueryNode::Not(expr) => {
                1.0 - self.estimate_selectivity(expr)
            },
            QueryNode::Group(expr) => {
                self.estimate_selectivity(expr)
            },
        }
    }

    fn term_to_string(&self, term: &TermQuery) -> String {
        let mut result = String::new();
        
        if let Some(field) = &term.field {
            result.push_str(&format!("{}:", field));
        }
        
        match &term.query_type {
            TermType::Phrase => result.push_str(&format!("\"{}\"", term.value)),
            TermType::Wildcard => result.push_str(&term.value),
            TermType::Range { start, end, inclusive } => {
                let bracket = if *inclusive { "[" } else { "{" };
                let bracket_end = if *inclusive { "]" } else { "}" };
                result.push_str(&format!("{}{} TO {}{}", bracket, start, end, bracket_end));
            },
            TermType::Exists => result.push_str(&format!("_exists_:{}", term.value)),
            TermType::Regex => result.push_str(&format!("/{}/", term.value)),
            TermType::Word => result.push_str(&term.value),
        }
        
        if let Some(boost) = term.boost {
            result.push_str(&format!("^{}", boost));
        }
        
        if let Some(fuzzy) = &term.fuzzy {
            result.push_str(&format!("~{}", fuzzy.distance));
        }
        
        result
    }

    // Optimization methods
    fn eliminate_double_negation(&self, tree: QueryNode, applied: &mut Vec<OptimizationType>) -> QueryNode {
        match tree {
            QueryNode::Not(expr) => {
                if let QueryNode::Not(inner) = *expr {
                    applied.push(OptimizationType::DoubleNegationElimination);
                    self.eliminate_double_negation(*inner, applied)
                } else {
                    QueryNode::Not(Box::new(self.eliminate_double_negation(*expr, applied)))
                }
            },
            QueryNode::Binary { left, operator, right } => {
                QueryNode::Binary {
                    left: Box::new(self.eliminate_double_negation(*left, applied)),
                    operator,
                    right: Box::new(self.eliminate_double_negation(*right, applied)),
                }
            },
            QueryNode::Group(expr) => {
                QueryNode::Group(Box::new(self.eliminate_double_negation(*expr, applied)))
            },
            other => other,
        }
    }

    fn constant_folding(&self, tree: QueryNode, applied: &mut Vec<OptimizationType>) -> QueryNode {
        // Simplified constant folding - remove empty nodes
        match tree {
            QueryNode::Binary { left, operator, right } => {
                let folded_left = self.constant_folding(*left, applied);
                let folded_right = self.constant_folding(*right, applied);
                
                match (&folded_left, &folded_right) {
                    (QueryNode::Empty, _) => {
                        applied.push(OptimizationType::ConstantFolding);
                        folded_right
                    },
                    (_, QueryNode::Empty) => {
                        applied.push(OptimizationType::ConstantFolding);
                        folded_left
                    },
                    _ => QueryNode::Binary {
                        left: Box::new(folded_left),
                        operator,
                        right: Box::new(folded_right),
                    }
                }
            },
            other => other,
        }
    }

    fn reorder_terms(&self, tree: QueryNode, applied: &mut Vec<OptimizationType>) -> QueryNode {
        // Reorder AND terms by estimated selectivity (most selective first)
        match tree {
            QueryNode::Binary { left, operator, right } => {
                if operator == BooleanOperator::And {
                    let left_sel = self.estimate_selectivity(&left);
                    let right_sel = self.estimate_selectivity(&right);
                    
                    if right_sel < left_sel {
                        applied.push(OptimizationType::TermReordering);
                        QueryNode::Binary {
                            left: Box::new(self.reorder_terms(*right, applied)),
                            operator,
                            right: Box::new(self.reorder_terms(*left, applied)),
                        }
                    } else {
                        QueryNode::Binary {
                            left: Box::new(self.reorder_terms(*left, applied)),
                            operator,
                            right: Box::new(self.reorder_terms(*right, applied)),
                        }
                    }
                } else {
                    QueryNode::Binary {
                        left: Box::new(self.reorder_terms(*left, applied)),
                        operator,
                        right: Box::new(self.reorder_terms(*right, applied)),
                    }
                }
            },
            other => other,
        }
    }

    fn push_not_down(&self, tree: QueryNode, _applied: &mut Vec<OptimizationType>) -> QueryNode {
        // Apply De Morgan's laws: NOT (A AND B) = (NOT A) OR (NOT B)
        // This is a simplified implementation
        tree
    }

    fn factor_common_terms(&self, tree: QueryNode, _applied: &mut Vec<OptimizationType>) -> QueryNode {
        // Factor out common terms in OR expressions
        // This is a simplified implementation
        tree
    }
}

impl QueryLexer {
    fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let current_char = chars.get(0).copied();
        
        Self {
            input: chars,
            position: 0,
            current_char,
        }
    }

    fn advance(&mut self) {
        self.position += 1;
        self.current_char = self.input.get(self.position).copied();
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.position + 1).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_word(&mut self) -> String {
        let mut word = String::new();
        
        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '_' || ch == '.' || ch == '-' || ch == '*' || ch == '?' {
                word.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        word
    }

    fn read_quoted_string(&mut self) -> String {
        let mut content = String::new();
        self.advance(); // Skip opening quote
        
        while let Some(ch) = self.current_char {
            if ch == '"' {
                self.advance(); // Skip closing quote
                break;
            } else if ch == '\\' {
                self.advance(); // Skip escape character
                if let Some(escaped) = self.current_char {
                    content.push(escaped);
                    self.advance();
                }
            } else {
                content.push(ch);
                self.advance();
            }
        }
        
        content
    }

    fn next_token(&mut self) -> Token {
        self.skip_whitespace();
        
        match self.current_char {
            None => Token::Eof,
            Some('(') => {
                self.advance();
                Token::LeftParen
            },
            Some(')') => {
                self.advance();
                Token::RightParen
            },
            Some('"') => {
                let content = self.read_quoted_string();
                Token::Phrase(content)
            },
            Some(ch) if ch.is_alphabetic() || ch == '_' => {
                let word = self.read_word();
                
                // Check if it's a field specifier (word:)
                if self.current_char == Some(':') {
                    self.advance(); // Skip ':'
                    self.skip_whitespace();
                    
                    // Read the field value
                    let value = if self.current_char == Some('"') {
                        self.read_quoted_string()
                    } else {
                        self.read_word()
                    };
                    
                    Token::FieldSpecifier { field: word, value }
                } else {
                    // Check if it's an operator
                    if let Some(op) = self.word_to_operator(&word) {
                        Token::Operator(op)
                    } else {
                        Token::Word(word)
                    }
                }
            },
            Some('*') | Some('?') => {
                let pattern = self.read_word();
                Token::Wildcard(pattern)
            },
            Some(_) => {
                // Handle other special characters
                let word = self.read_word();
                if let Some(op) = self.word_to_operator(&word) {
                    Token::Operator(op)
                } else {
                    Token::Word(word)
                }
            },
        }
    }

    fn word_to_operator(&self, word: &str) -> Option<BooleanOperator> {
        match word.to_uppercase().as_str() {
            "AND" | "&" | "&&" => Some(BooleanOperator::And),
            "OR" | "|" | "||" => Some(BooleanOperator::Or),
            "NOT" | "!" | "-" => Some(BooleanOperator::Not),
            _ => None,
        }
    }
}

impl QueryParser {
    fn new(
        mut lexer: QueryLexer,
        field_aliases: &HashMap<String, String>,
        operator_aliases: &HashMap<String, BooleanOperator>,
    ) -> Self {
        let current_token = lexer.next_token();
        
        Self {
            lexer,
            current_token,
            field_aliases: field_aliases.clone(),
            operator_aliases: operator_aliases.clone(),
        }
    }

    fn advance_token(&mut self) {
        self.current_token = self.lexer.next_token();
    }

    /// Parse the main expression with operator precedence
    fn parse_expression(&mut self) -> Result<QueryNode> {
        self.parse_or_expression()
    }

    /// Parse OR expressions (lowest precedence)
    fn parse_or_expression(&mut self) -> Result<QueryNode> {
        let mut left = self.parse_and_expression()?;

        while matches!(self.current_token, Token::Operator(BooleanOperator::Or)) {
            self.advance_token();
            let right = self.parse_and_expression()?;
            left = QueryNode::Binary {
                left: Box::new(left),
                operator: BooleanOperator::Or,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse AND expressions (medium precedence)
    fn parse_and_expression(&mut self) -> Result<QueryNode> {
        let mut left = self.parse_not_expression()?;

        while matches!(self.current_token, Token::Operator(BooleanOperator::And)) {
            self.advance_token();
            let right = self.parse_not_expression()?;
            left = QueryNode::Binary {
                left: Box::new(left),
                operator: BooleanOperator::And,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse NOT expressions (highest precedence)
    fn parse_not_expression(&mut self) -> Result<QueryNode> {
        if matches!(self.current_token, Token::Operator(BooleanOperator::Not)) {
            self.advance_token();
            let expr = self.parse_primary()?;
            Ok(QueryNode::Not(Box::new(expr)))
        } else {
            self.parse_primary()
        }
    }

    /// Parse primary expressions (terms, groups, etc.)
    fn parse_primary(&mut self) -> Result<QueryNode> {
        match &self.current_token.clone() {
            Token::LeftParen => {
                self.advance_token();
                let expr = self.parse_expression()?;
                
                if matches!(self.current_token, Token::RightParen) {
                    self.advance_token();
                    Ok(QueryNode::Group(Box::new(expr)))
                } else {
                    Ok(expr) // Missing closing paren, but continue
                }
            },
            Token::Word(word) => {
                let term = TermQuery {
                    field: None,
                    value: word.clone(),
                    query_type: TermType::Word,
                    boost: None,
                    fuzzy: None,
                    proximity: None,
                };
                self.advance_token();
                Ok(QueryNode::Term(term))
            },
            Token::Phrase(phrase) => {
                let term = TermQuery {
                    field: None,
                    value: phrase.clone(),
                    query_type: TermType::Phrase,
                    boost: None,
                    fuzzy: None,
                    proximity: None,
                };
                self.advance_token();
                Ok(QueryNode::Term(term))
            },
            Token::FieldSpecifier { field, value } => {
                let resolved_field = self.field_aliases.get(field)
                    .unwrap_or(field)
                    .clone();
                
                let term = TermQuery {
                    field: Some(resolved_field),
                    value: value.clone(),
                    query_type: if value.contains('*') || value.contains('?') {
                        TermType::Wildcard
                    } else {
                        TermType::Word
                    },
                    boost: None,
                    fuzzy: None,
                    proximity: None,
                };
                self.advance_token();
                Ok(QueryNode::Term(term))
            },
            Token::Wildcard(pattern) => {
                let term = TermQuery {
                    field: None,
                    value: pattern.clone(),
                    query_type: TermType::Wildcard,
                    boost: None,
                    fuzzy: None,
                    proximity: None,
                };
                self.advance_token();
                Ok(QueryNode::Term(term))
            },
            Token::Eof => Ok(QueryNode::Empty),
            _ => {
                // Unexpected token, skip and continue
                self.advance_token();
                self.parse_primary()
            }
        }
    }
}

impl Default for BooleanQueryParser {
    fn default() -> Self {
        Self::new()
    }
}

impl BooleanOperator {
    pub fn precedence(&self) -> u8 {
        match self {
            BooleanOperator::Or => 1,
            BooleanOperator::And => 2,
            BooleanOperator::Not => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_term_parsing() {
        let parser = BooleanQueryParser::new();
        let result = parser.parse("hello").unwrap();
        
        match result.tree {
            QueryNode::Term(term) => {
                assert_eq!(term.value, "hello");
                assert_eq!(term.query_type, TermType::Word);
            },
            _ => panic!("Expected term node"),
        }
    }

    #[test]
    fn test_and_expression() {
        let parser = BooleanQueryParser::new();
        let result = parser.parse("hello AND world").unwrap();
        
        match result.tree {
            QueryNode::Binary { operator, .. } => {
                assert_eq!(operator, BooleanOperator::And);
            },
            _ => panic!("Expected binary node"),
        }
    }

    #[test]
    fn test_field_specifier() {
        let parser = BooleanQueryParser::new();
        let result = parser.parse("author:john").unwrap();
        
        match result.tree {
            QueryNode::Term(term) => {
                assert_eq!(term.field, Some("creator".to_string())); // Should be aliased
                assert_eq!(term.value, "john");
            },
            _ => panic!("Expected term node"),
        }
    }

    #[test]
    fn test_phrase_query() {
        let parser = BooleanQueryParser::new();
        let result = parser.parse("\"hello world\"").unwrap();
        
        match result.tree {
            QueryNode::Term(term) => {
                assert_eq!(term.value, "hello world");
                assert_eq!(term.query_type, TermType::Phrase);
            },
            _ => panic!("Expected term node"),
        }
    }

    #[test]
    fn test_complex_expression() {
        let parser = BooleanQueryParser::new();
        let result = parser.parse("(author:john OR author:jane) AND type:pdf").unwrap();
        
        // Should parse without errors
        assert!(result.complexity_score > 1.0);
    }
}