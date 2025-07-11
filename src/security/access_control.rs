use std::collections::HashMap;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use super::{SecurityConfig, SecurityContext, Permission};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPolicy {
    pub id: String,
    pub name: String,
    pub description: String,
    pub rules: Vec<AccessRule>,
    pub priority: u32,
    pub enabled: bool,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRule {
    pub id: String,
    pub resource_pattern: String,
    pub action: String,
    pub effect: Effect,
    pub conditions: Vec<Condition>,
    pub time_restrictions: Option<TimeRestriction>,
    pub ip_restrictions: Option<IpRestriction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Effect {
    Allow,
    Deny,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub field: String,
    pub operator: ConditionOperator,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    Contains,
    StartsWith,
    EndsWith,
    GreaterThan,
    LessThan,
    MatchesRegex,
    In,
    NotIn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRestriction {
    pub allowed_hours: Vec<u8>, // 0-23
    pub allowed_days: Vec<u8>,  // 0-6 (Sunday-Saturday)
    pub timezone: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpRestriction {
    pub allowed_ips: Vec<String>,
    pub blocked_ips: Vec<String>,
    pub allowed_networks: Vec<String>,
    pub blocked_networks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub id: String,
    pub name: String,
    pub description: String,
    pub permissions: Vec<Permission>,
    pub policies: Vec<String>, // Policy IDs
    pub inherits_from: Vec<String>, // Role IDs
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub username: String,
    pub email: Option<String>,
    pub roles: Vec<String>, // Role IDs
    pub direct_permissions: Vec<Permission>,
    pub metadata: HashMap<String, String>,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub last_login: Option<SystemTime>,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRequest {
    pub user_id: Option<String>,
    pub session_id: String,
    pub resource: String,
    pub action: String,
    pub context: AccessContext,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessContext {
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub request_id: Option<String>,
    pub additional_attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessDecision {
    pub decision: Decision,
    pub reason: String,
    pub applied_policies: Vec<String>,
    pub applied_rules: Vec<String>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Decision {
    Allow,
    Deny,
    NotApplicable,
}

pub struct AccessController {
    config: SecurityConfig,
    policies: HashMap<String, AccessPolicy>,
    roles: HashMap<String, Role>,
    users: HashMap<String, User>,
    cache: HashMap<String, AccessDecision>,
    cache_ttl: std::time::Duration,
}

impl AccessController {
    pub fn new(config: &SecurityConfig) -> Result<Self> {
        let mut controller = Self {
            config: config.clone(),
            policies: HashMap::new(),
            roles: HashMap::new(),
            users: HashMap::new(),
            cache: HashMap::new(),
            cache_ttl: std::time::Duration::from_secs(300), // 5 minutes
        };

        controller.initialize_default_policies()?;
        controller.initialize_default_roles()?;

        Ok(controller)
    }

    pub fn check_permission(&self, context: &SecurityContext, resource: &str, action: &str) -> Result<bool> {
        let request = AccessRequest {
            user_id: context.user_id.clone(),
            session_id: context.session_id.clone(),
            resource: resource.to_string(),
            action: action.to_string(),
            context: AccessContext {
                ip_address: context.ip_address.clone(),
                user_agent: context.user_agent.clone(),
                request_id: None,
                additional_attributes: HashMap::new(),
            },
            timestamp: SystemTime::now(),
        };

        let decision = self.evaluate_access_request(&request)?;
        Ok(matches!(decision.decision, Decision::Allow))
    }

    pub fn evaluate_access_request(&self, request: &AccessRequest) -> Result<AccessDecision> {
        // Check cache first
        let cache_key = format!("{}:{}:{}", request.session_id, request.resource, request.action);
        if let Some(cached_decision) = self.cache.get(&cache_key) {
            if cached_decision.timestamp.elapsed().unwrap_or(std::time::Duration::from_secs(0)) < self.cache_ttl {
                return Ok(cached_decision.clone());
            }
        }

        // Evaluate policies
        let mut decision = AccessDecision {
            decision: Decision::Deny,
            reason: "Default deny".to_string(),
            applied_policies: Vec::new(),
            applied_rules: Vec::new(),
            timestamp: SystemTime::now(),
        };

        // Get applicable policies
        let mut applicable_policies: Vec<&AccessPolicy> = self.policies.values()
            .filter(|policy| policy.enabled)
            .collect();

        // Sort by priority (higher priority first)
        applicable_policies.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Evaluate each policy
        for policy in applicable_policies {
            let policy_decision = self.evaluate_policy(policy, request)?;
            
            if matches!(policy_decision.decision, Decision::Allow | Decision::Deny) {
                decision = policy_decision;
                decision.applied_policies.push(policy.id.clone());
                
                // Stop on first definitive decision (allow or deny)
                if matches!(decision.decision, Decision::Deny) {
                    break;
                }
            }
        }

        // Check direct permissions if no policy decision
        if matches!(decision.decision, Decision::NotApplicable) {
            if let Some(user_id) = &request.user_id {
                decision = self.check_direct_permissions(user_id, &request.resource, &request.action)?;
            }
        }

        // Cache the decision
        // Note: In a real implementation, you'd want to use a proper cache with expiration
        
        Ok(decision)
    }

    pub fn create_policy(&mut self, policy: AccessPolicy) -> Result<()> {
        self.policies.insert(policy.id.clone(), policy);
        Ok(())
    }

    pub fn update_policy(&mut self, policy_id: &str, policy: AccessPolicy) -> Result<()> {
        if self.policies.contains_key(policy_id) {
            self.policies.insert(policy_id.to_string(), policy);
            Ok(())
        } else {
            Err(crate::error::AppError::NotFound("Policy not found".to_string()))
        }
    }

    pub fn delete_policy(&mut self, policy_id: &str) -> Result<()> {
        if self.policies.remove(policy_id).is_some() {
            Ok(())
        } else {
            Err(crate::error::AppError::NotFound("Policy not found".to_string()))
        }
    }

    pub fn create_role(&mut self, role: Role) -> Result<()> {
        self.roles.insert(role.id.clone(), role);
        Ok(())
    }

    pub fn update_role(&mut self, role_id: &str, role: Role) -> Result<()> {
        if self.roles.contains_key(role_id) {
            self.roles.insert(role_id.to_string(), role);
            Ok(())
        } else {
            Err(crate::error::AppError::NotFound("Role not found".to_string()))
        }
    }

    pub fn delete_role(&mut self, role_id: &str) -> Result<()> {
        if self.roles.remove(role_id).is_some() {
            Ok(())
        } else {
            Err(crate::error::AppError::NotFound("Role not found".to_string()))
        }
    }

    pub fn create_user(&mut self, user: User) -> Result<()> {
        self.users.insert(user.id.clone(), user);
        Ok(())
    }

    pub fn update_user(&mut self, user_id: &str, user: User) -> Result<()> {
        if self.users.contains_key(user_id) {
            self.users.insert(user_id.to_string(), user);
            Ok(())
        } else {
            Err(crate::error::AppError::NotFound("User not found".to_string()))
        }
    }

    pub fn delete_user(&mut self, user_id: &str) -> Result<()> {
        if self.users.remove(user_id).is_some() {
            Ok(())
        } else {
            Err(crate::error::AppError::NotFound("User not found".to_string()))
        }
    }

    pub fn assign_role_to_user(&mut self, user_id: &str, role_id: &str) -> Result<()> {
        if let Some(user) = self.users.get_mut(user_id) {
            if !user.roles.contains(&role_id.to_string()) {
                user.roles.push(role_id.to_string());
                user.updated_at = SystemTime::now();
            }
            Ok(())
        } else {
            Err(crate::error::AppError::NotFound("User not found".to_string()))
        }
    }

    pub fn remove_role_from_user(&mut self, user_id: &str, role_id: &str) -> Result<()> {
        if let Some(user) = self.users.get_mut(user_id) {
            user.roles.retain(|r| r != role_id);
            user.updated_at = SystemTime::now();
            Ok(())
        } else {
            Err(crate::error::AppError::NotFound("User not found".to_string()))
        }
    }

    pub fn get_user_permissions(&self, user_id: &str) -> Result<Vec<Permission>> {
        let user = self.users.get(user_id).ok_or_else(|| {
            crate::error::AppError::NotFound("User not found".to_string())
        })?;

        let mut permissions = user.direct_permissions.clone();

        // Add permissions from roles
        for role_id in &user.roles {
            if let Some(role) = self.roles.get(role_id) {
                permissions.extend(role.permissions.clone());
                
                // Recursively get permissions from inherited roles
                let inherited_permissions = self.get_inherited_permissions(role)?;
                permissions.extend(inherited_permissions);
            }
        }

        // Remove duplicates
        permissions.dedup_by(|a, b| a.resource == b.resource && a.action == b.action);

        Ok(permissions)
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    fn evaluate_policy(&self, policy: &AccessPolicy, request: &AccessRequest) -> Result<AccessDecision> {
        let mut decision = AccessDecision {
            decision: Decision::NotApplicable,
            reason: "No matching rules".to_string(),
            applied_policies: Vec::new(),
            applied_rules: Vec::new(),
            timestamp: SystemTime::now(),
        };

        for rule in &policy.rules {
            if self.rule_matches(rule, request)? {
                decision.decision = match rule.effect {
                    Effect::Allow => Decision::Allow,
                    Effect::Deny => Decision::Deny,
                };
                decision.reason = format!("Rule {} applied", rule.id);
                decision.applied_rules.push(rule.id.clone());
                
                // Stop on first matching rule
                break;
            }
        }

        Ok(decision)
    }

    fn rule_matches(&self, rule: &AccessRule, request: &AccessRequest) -> Result<bool> {
        // Check resource pattern
        if !self.matches_pattern(&rule.resource_pattern, &request.resource)? {
            return Ok(false);
        }

        // Check action
        if rule.action != "*" && rule.action != request.action {
            return Ok(false);
        }

        // Check conditions
        for condition in &rule.conditions {
            if !self.evaluate_condition(condition, request)? {
                return Ok(false);
            }
        }

        // Check time restrictions
        if let Some(time_restriction) = &rule.time_restrictions {
            if !self.check_time_restriction(time_restriction, request)? {
                return Ok(false);
            }
        }

        // Check IP restrictions
        if let Some(ip_restriction) = &rule.ip_restrictions {
            if !self.check_ip_restriction(ip_restriction, request)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn matches_pattern(&self, pattern: &str, resource: &str) -> Result<bool> {
        if pattern == "*" {
            return Ok(true);
        }

        // Convert glob pattern to regex
        let regex_pattern = pattern
            .replace("*", ".*")
            .replace("?", ".");

        let regex = regex::Regex::new(&format!("^{}$", regex_pattern))?;
        Ok(regex.is_match(resource))
    }

    fn evaluate_condition(&self, condition: &Condition, request: &AccessRequest) -> Result<bool> {
        let field_value = match condition.field.as_str() {
            "user_id" => request.user_id.as_deref().unwrap_or(""),
            "session_id" => &request.session_id,
            "resource" => &request.resource,
            "action" => &request.action,
            "ip_address" => request.context.ip_address.as_deref().unwrap_or(""),
            "user_agent" => request.context.user_agent.as_deref().unwrap_or(""),
            _ => request.context.additional_attributes.get(&condition.field).map(|s| s.as_str()).unwrap_or(""),
        };

        match condition.operator {
            ConditionOperator::Equals => Ok(field_value == condition.value),
            ConditionOperator::NotEquals => Ok(field_value != condition.value),
            ConditionOperator::Contains => Ok(field_value.contains(&condition.value)),
            ConditionOperator::StartsWith => Ok(field_value.starts_with(&condition.value)),
            ConditionOperator::EndsWith => Ok(field_value.ends_with(&condition.value)),
            ConditionOperator::GreaterThan => {
                let field_num: f64 = field_value.parse().unwrap_or(0.0);
                let condition_num: f64 = condition.value.parse().unwrap_or(0.0);
                Ok(field_num > condition_num)
            }
            ConditionOperator::LessThan => {
                let field_num: f64 = field_value.parse().unwrap_or(0.0);
                let condition_num: f64 = condition.value.parse().unwrap_or(0.0);
                Ok(field_num < condition_num)
            }
            ConditionOperator::MatchesRegex => {
                let regex = regex::Regex::new(&condition.value)?;
                Ok(regex.is_match(field_value))
            }
            ConditionOperator::In => {
                let values: Vec<&str> = condition.value.split(',').collect();
                Ok(values.contains(&field_value))
            }
            ConditionOperator::NotIn => {
                let values: Vec<&str> = condition.value.split(',').collect();
                Ok(!values.contains(&field_value))
            }
        }
    }

    fn check_time_restriction(&self, restriction: &TimeRestriction, request: &AccessRequest) -> Result<bool> {
        use chrono::{DateTime, Utc, Timelike, Weekday};
        
        let now = DateTime::<Utc>::from(request.timestamp);
        let hour = now.hour() as u8;
        let weekday = now.weekday().number_from_sunday() as u8 - 1; // Convert to 0-6

        if !restriction.allowed_hours.is_empty() && !restriction.allowed_hours.contains(&hour) {
            return Ok(false);
        }

        if !restriction.allowed_days.is_empty() && !restriction.allowed_days.contains(&weekday) {
            return Ok(false);
        }

        Ok(true)
    }

    fn check_ip_restriction(&self, restriction: &IpRestriction, request: &AccessRequest) -> Result<bool> {
        let ip_address = match &request.context.ip_address {
            Some(ip) => ip,
            None => return Ok(false),
        };

        // Check blocked IPs first
        if restriction.blocked_ips.contains(ip_address) {
            return Ok(false);
        }

        // Check blocked networks
        for network in &restriction.blocked_networks {
            if self.ip_in_network(ip_address, network)? {
                return Ok(false);
            }
        }

        // Check allowed IPs
        if !restriction.allowed_ips.is_empty() {
            if !restriction.allowed_ips.contains(ip_address) {
                // Check allowed networks
                let mut allowed = false;
                for network in &restriction.allowed_networks {
                    if self.ip_in_network(ip_address, network)? {
                        allowed = true;
                        break;
                    }
                }
                return Ok(allowed);
            }
        }

        Ok(true)
    }

    fn ip_in_network(&self, ip: &str, network: &str) -> Result<bool> {
        // Simplified IP network check
        // In a real implementation, use a proper CIDR library
        if network.contains('/') {
            let parts: Vec<&str> = network.split('/').collect();
            if parts.len() == 2 {
                let network_ip = parts[0];
                let prefix_len: u32 = parts[1].parse().unwrap_or(32);
                
                // For simplicity, just check if IP starts with network IP
                if prefix_len >= 24 {
                    return Ok(ip.starts_with(network_ip));
                }
            }
        }
        
        Ok(false)
    }

    fn check_direct_permissions(&self, user_id: &str, resource: &str, action: &str) -> Result<AccessDecision> {
        let permissions = self.get_user_permissions(user_id)?;
        
        for permission in permissions {
            if self.matches_pattern(&permission.resource, resource)? && 
               (permission.action == "*" || permission.action == action) {
                
                if permission.granted {
                    // Check if permission has expired
                    if let Some(expires_at) = permission.expires_at {
                        if SystemTime::now() > expires_at {
                            continue;
                        }
                    }
                    
                    return Ok(AccessDecision {
                        decision: Decision::Allow,
                        reason: "Direct permission granted".to_string(),
                        applied_policies: Vec::new(),
                        applied_rules: Vec::new(),
                        timestamp: SystemTime::now(),
                    });
                }
            }
        }

        Ok(AccessDecision {
            decision: Decision::Deny,
            reason: "No matching permissions".to_string(),
            applied_policies: Vec::new(),
            applied_rules: Vec::new(),
            timestamp: SystemTime::now(),
        })
    }

    fn get_inherited_permissions(&self, role: &Role) -> Result<Vec<Permission>> {
        let mut permissions = Vec::new();

        for parent_role_id in &role.inherits_from {
            if let Some(parent_role) = self.roles.get(parent_role_id) {
                permissions.extend(parent_role.permissions.clone());
                
                // Recursively get permissions from inherited roles
                let inherited = self.get_inherited_permissions(parent_role)?;
                permissions.extend(inherited);
            }
        }

        Ok(permissions)
    }

    fn initialize_default_policies(&mut self) -> Result<()> {
        // Create a default allow policy for basic operations
        let default_policy = AccessPolicy {
            id: "default-allow".to_string(),
            name: "Default Allow Policy".to_string(),
            description: "Default policy that allows basic operations".to_string(),
            rules: vec![
                AccessRule {
                    id: "allow-search".to_string(),
                    resource_pattern: "search/*".to_string(),
                    action: "read".to_string(),
                    effect: Effect::Allow,
                    conditions: Vec::new(),
                    time_restrictions: None,
                    ip_restrictions: None,
                },
                AccessRule {
                    id: "allow-files".to_string(),
                    resource_pattern: "files/*".to_string(),
                    action: "read".to_string(),
                    effect: Effect::Allow,
                    conditions: Vec::new(),
                    time_restrictions: None,
                    ip_restrictions: None,
                },
            ],
            priority: 1,
            enabled: true,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
        };

        self.create_policy(default_policy)?;

        // Create a security policy that blocks dangerous operations
        let security_policy = AccessPolicy {
            id: "security-restrictions".to_string(),
            name: "Security Restrictions".to_string(),
            description: "Policy that blocks potentially dangerous operations".to_string(),
            rules: vec![
                AccessRule {
                    id: "block-system-files".to_string(),
                    resource_pattern: "/etc/*".to_string(),
                    action: "*".to_string(),
                    effect: Effect::Deny,
                    conditions: Vec::new(),
                    time_restrictions: None,
                    ip_restrictions: None,
                },
                AccessRule {
                    id: "block-executables".to_string(),
                    resource_pattern: "*.exe".to_string(),
                    action: "*".to_string(),
                    effect: Effect::Deny,
                    conditions: Vec::new(),
                    time_restrictions: None,
                    ip_restrictions: None,
                },
            ],
            priority: 100,
            enabled: true,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
        };

        self.create_policy(security_policy)?;

        Ok(())
    }

    fn initialize_default_roles(&mut self) -> Result<()> {
        // Create a basic user role
        let user_role = Role {
            id: "user".to_string(),
            name: "User".to_string(),
            description: "Basic user role with read access".to_string(),
            permissions: vec![
                Permission {
                    resource: "search/*".to_string(),
                    action: "read".to_string(),
                    granted: true,
                    expires_at: None,
                },
                Permission {
                    resource: "files/*".to_string(),
                    action: "read".to_string(),
                    granted: true,
                    expires_at: None,
                },
            ],
            policies: vec!["default-allow".to_string()],
            inherits_from: Vec::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
        };

        self.create_role(user_role)?;

        // Create an admin role
        let admin_role = Role {
            id: "admin".to_string(),
            name: "Administrator".to_string(),
            description: "Administrator role with full access".to_string(),
            permissions: vec![
                Permission {
                    resource: "*".to_string(),
                    action: "*".to_string(),
                    granted: true,
                    expires_at: None,
                },
            ],
            policies: vec!["default-allow".to_string()],
            inherits_from: vec!["user".to_string()],
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
        };

        self.create_role(admin_role)?;

        Ok(())
    }
}