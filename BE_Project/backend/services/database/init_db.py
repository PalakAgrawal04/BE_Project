"""
Database initialization script.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from mysql.connector import connect, Error

logger = logging.getLogger(__name__)

def init_database(sql_path: Optional[str] = None) -> bool:
    """
    Initialize database with required tables.
    
    Args:
        sql_path: Path to SQL schema file. If None, uses default schema.sql
        
    Returns:
        True if initialization successful, False otherwise
    """
    if sql_path is None:
        sql_path = Path(__file__).parent / 'schema.sql'
        
    try:
        # Connect with admin privileges
        conn = connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', '')
        )
        
        cursor = conn.cursor()
        
        # Create database if not exists
        db_name = os.getenv('MYSQL_DATABASE', 'intelliquery')
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        
        # Create read-only user if not exists
        ro_user = os.getenv('MYSQL_USER', 'intelliquery')
        ro_pass = os.getenv('MYSQL_PASSWORD', '')
        
        cursor.execute(f"CREATE USER IF NOT EXISTS '{ro_user}'@'localhost' "
                      f"IDENTIFIED BY '{ro_pass}'")
                      
        cursor.execute(f"GRANT SELECT ON {db_name}.* TO '{ro_user}'@'localhost'")
        
        # Switch to application database
        cursor.execute(f"USE {db_name}")
        
        # Read and execute schema file
        with open(sql_path) as f:
            schema = f.read()
            
        # Execute each statement
        for statement in schema.split(';'):
            if statement.strip():
                cursor.execute(statement)
                
        conn.commit()
        logger.info("Database initialized successfully")
        return True
        
    except Error as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False
        
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database
    init_database()