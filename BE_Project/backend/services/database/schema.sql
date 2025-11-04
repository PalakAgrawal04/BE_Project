# Schema for query analytics table
CREATE TABLE IF NOT EXISTS query_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_type VARCHAR(50) NOT NULL,
    execution_time_ms FLOAT NOT NULL,
    status VARCHAR(50) NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_query_type (query_type),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

# Schema for query metrics
CREATE TABLE IF NOT EXISTS query_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    labels JSON,
    INDEX idx_metric_name (metric_name),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;