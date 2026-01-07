-- PostgreSQL Setup Script for SQL Compiler Agent
-- This script creates a read-only user with appropriate permissions

-- Create read-only user for AI agent
-- Note: Replace 'secure_password' with a strong password
CREATE USER ai_agent_readonly WITH PASSWORD 'secure_password';

-- Grant CONNECT privilege on the database
-- Note: Replace 'your_database' with your actual database name
GRANT CONNECT ON DATABASE your_database TO ai_agent_readonly;

-- Grant USAGE on the public schema
GRANT USAGE ON SCHEMA public TO ai_agent_readonly;

-- Grant SELECT on all existing tables in public schema
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ai_agent_readonly;

-- Ensure future tables are also readable
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
GRANT SELECT ON TABLES TO ai_agent_readonly;

-- Grant SELECT on all existing sequences (for auto-increment columns)
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO ai_agent_readonly;

-- Ensure future sequences are also readable
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
GRANT SELECT ON SEQUENCES TO ai_agent_readonly;

-- Explicitly revoke dangerous permissions
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public FROM ai_agent_readonly;
REVOKE CREATE, USAGE ON ALL SEQUENCES IN SCHEMA public FROM ai_agent_readonly;

-- Revoke any potential dangerous privileges
REVOKE ALL ON DATABASE your_database FROM ai_agent_readonly;
GRANT CONNECT ON DATABASE your_database TO ai_agent_readonly;

-- Optional: Create a more restricted user if needed
-- This user can only read specific tables
-- CREATE USER ai_agent_restricted WITH PASSWORD 'secure_password_2';
-- GRANT CONNECT ON DATABASE your_database TO ai_agent_restricted;
-- GRANT USAGE ON SCHEMA public TO ai_agent_restricted;
-- GRANT SELECT ON TABLE specific_table1, specific_table2 TO ai_agent_restricted;

