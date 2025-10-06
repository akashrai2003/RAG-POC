"""
SQL service for executing queries on asset data.
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re

from config.settings import config, QueryConfig
from models.schemas import AssetRecord


class SQLService:
    """Service for handling SQL queries on asset data."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.sqlite_db_path
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=config.google_api_key,
            temperature=QueryConfig.SQL_GENERATION_TEMPERATURE
        )
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with asset table."""
        try:
            # Ensure directory exists before creating database
            import os
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                print(f"✅ Created database directory: {db_dir}")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create assets table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS assets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        asset_id TEXT UNIQUE NOT NULL,
                        account_name TEXT,
                        action_needed TEXT,
                        battery_voltage REAL,
                        cardinal_tag BOOLEAN,
                        current_location TEXT,
                        date_shipped DATETIME,
                        est_battery_calculate REAL,
                        last_connected DATETIME,
                        power_reset_occurred BOOLEAN,
                        power_reset_time DATETIME,
                        powerup_time INTEGER,
                        product_name TEXT,
                        state_of_pallet TEXT,
                        account_address TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for common query patterns
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_asset_id ON assets(asset_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_battery_voltage ON assets(battery_voltage)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_product_name ON assets(product_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_state_of_pallet ON assets(state_of_pallet)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_date_shipped ON assets(date_shipped)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_connected ON assets(last_connected)")
                
                conn.commit()
                
        except Exception as e:
            raise Exception(f"Failed to initialize database: {str(e)}")
    
    def insert_assets(self, assets: List[AssetRecord]) -> Dict[str, Any]:
        """Insert asset records into the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                inserted_count = 0
                updated_count = 0
                errors = []
                
                for asset in assets:
                    try:
                        # Prepare data for insertion
                        data = {
                            'asset_id': asset.asset_id,
                            'account_name': asset.account_name,
                            'action_needed': asset.action_needed,
                            'battery_voltage': asset.battery_voltage,
                            'cardinal_tag': asset.cardinal_tag,
                            'current_location': asset.current_location,
                            'date_shipped': asset.date_shipped.isoformat() if asset.date_shipped else None,
                            'est_battery_calculate': asset.est_battery_calculate,
                            'last_connected': asset.last_connected.isoformat() if asset.last_connected else None,
                            'power_reset_occurred': asset.power_reset_occurred,
                            'power_reset_time': asset.power_reset_time.isoformat() if asset.power_reset_time else None,
                            'powerup_time': asset.powerup_time,
                            'product_name': asset.product_name,
                            'state_of_pallet': asset.state_of_pallet,
                            'account_address': asset.account_address
                        }
                        
                        # Try to insert, update if exists
                        cursor.execute("""
                            INSERT OR REPLACE INTO assets (
                                asset_id, account_name, action_needed, battery_voltage,
                                cardinal_tag, current_location, date_shipped, est_battery_calculate,
                                last_connected, power_reset_occurred, power_reset_time,
                                powerup_time, product_name, state_of_pallet, account_address,
                                updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, tuple(data.values()))
                        
                        if cursor.rowcount == 1:
                            inserted_count += 1
                        else:
                            updated_count += 1
                            
                    except Exception as e:
                        errors.append(f"Error inserting asset {asset.asset_id}: {str(e)}")
                
                conn.commit()
                
                return {
                    'success': True,
                    'inserted': inserted_count,
                    'updated': updated_count,
                    'errors': errors
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'inserted': 0,
                'updated': 0,
                'errors': []
            }
    
    def execute_natural_language_query(self, query_description: str) -> Dict[str, Any]:
        """Convert natural language query to SQL and execute."""
        try:
            print(f"\n{'='*80}")
            print(f"🔍 SQL SERVICE: Processing query")
            print(f"📝 Query: {query_description}")
            print(f"{'='*80}")
            
            # Get database schema
            schema_info = self._get_schema_info()
            
            # Generate SQL query using LLM
            print(f"🤖 Generating SQL query using Gemini...")
            sql_query = self._generate_sql_query(query_description, schema_info)
            
            if not sql_query:
                print(f"❌ Failed to generate SQL query")
                return {
                    'success': False,
                    'error': 'Could not generate SQL query'
                }
            
            print(f"✅ Generated SQL Query:")
            print(f"   {sql_query}")
            
            # Execute the query
            print(f"⚡ Executing SQL query...")
            result = self._execute_sql_query(sql_query)
            result['sql_query'] = sql_query
            
            if result.get('success'):
                row_count = len(result.get('data', []))
                print(f"✅ Query executed successfully - {row_count} rows returned")
            else:
                print(f"❌ Query execution failed: {result.get('error')}")
            
            print(f"{'='*80}\n")
            return result
            
        except Exception as e:
            print(f"❌ SQL SERVICE ERROR: {str(e)}")
            print(f"{'='*80}\n")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_sql_query(self, query_description: str, schema_info: str) -> Optional[str]:
        """Generate SQL query from natural language description."""
        prompt = f"""
        You are an expert SQL query generator for an asset tracking database.
        
        Database Schema:
        {schema_info}
        
        User Request: {query_description}
        
        CRITICAL RULES - Read the schema carefully:
        
        1. **EXACT COLUMN NAMES** - Use ONLY these columns (case-sensitive):
           - asset_id (NOT Asset_ID or id)
           - account_name (NOT account)
           - action_needed
           - battery_voltage (NOT voltage)
           - cardinal_tag
           - current_location (NOT location)
           - date_shipped
           - est_battery_calculate
           - last_connected
           - power_reset_occurred
           - power_reset_time
           - powerup_time
           - product_name (NOT product)
           - state_of_pallet (NOT state, NOT status)
           - account_address
        
        2. Common user terms mapping:
           - "state" or "status" → use state_of_pallet
           - "location" → use current_location
           - "voltage" → use battery_voltage
           - "product" → use product_name
           - "account" → use account_name
        
        3. For string matching:
           - Use LIKE with '%value%' for partial matches
           - String comparisons are case-insensitive with LIKE
           - Example: state_of_pallet LIKE '%In Network%'
        
        4. Other rules:
           - For boolean values, use 1 for true and 0 for false
           - Limit results to 100 rows unless specifically asked for more
           - Use proper numerical operators for voltage comparisons
        
        Return ONLY the SQL query, no explanations or markdown formatting.
        
        Examples:
        - "assets with battery voltage less than 6" 
          → SELECT * FROM assets WHERE battery_voltage < 6 LIMIT 100
          
        - "count assets by state" 
          → SELECT state_of_pallet, COUNT(*) as count FROM assets GROUP BY state_of_pallet
          
        - "assets with low voltage in network state"
          → SELECT * FROM assets WHERE battery_voltage < 4.5 AND state_of_pallet LIKE '%In Network%' LIMIT 100
        
        SQL Query:
        """
        
        try:
            response = self.llm.invoke(prompt)
            sql_query = response.content.strip()
            
            # Clean up the response (remove any markdown formatting)
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            sql_query = sql_query.strip()
            
            # Basic validation
            if not sql_query.upper().startswith('SELECT'):
                return None
            
            # Prevent dangerous operations
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
            if any(keyword in sql_query.upper() for keyword in dangerous_keywords):
                return None
            
            return sql_query
            
        except Exception as e:
            print(f"Error generating SQL query: {e}")
            return None
    
    def _execute_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query and return results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Use row factory to get column names
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                data = [dict(row) for row in rows]
                
                return {
                    'success': True,
                    'data': data,
                    'row_count': len(data)
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': []
            }
    
    def _get_schema_info(self) -> str:
        """Get database schema information."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get table schema
                cursor.execute("PRAGMA table_info(assets)")
                columns = cursor.fetchall()
                
                # Get sample data for context
                cursor.execute("SELECT * FROM assets LIMIT 3")
                sample_rows = cursor.fetchall()
                
                # Format schema information
                schema_info = "Assets Table Schema:\n"
                schema_info += "Columns:\n"
                
                for col in columns:
                    schema_info += f"- {col[1]} ({col[2]})\n"
                
                if sample_rows:
                    schema_info += "\nSample Data (first 3 rows):\n"
                    for i, row in enumerate(sample_rows, 1):
                        schema_info += f"Row {i}: {dict(row)}\n"
                
                return schema_info
                
        except Exception as e:
            return f"Could not retrieve schema: {str(e)}"
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total records
                cursor.execute("SELECT COUNT(*) FROM assets")
                total_records = cursor.fetchone()[0]
                
                # Records by state
                cursor.execute("""
                    SELECT state_of_pallet, COUNT(*) as count 
                    FROM assets 
                    WHERE state_of_pallet IS NOT NULL 
                    GROUP BY state_of_pallet
                """)
                state_distribution = dict(cursor.fetchall())
                
                # Battery voltage stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as count,
                        AVG(battery_voltage) as avg_voltage,
                        MIN(battery_voltage) as min_voltage,
                        MAX(battery_voltage) as max_voltage
                    FROM assets 
                    WHERE battery_voltage IS NOT NULL
                """)
                voltage_stats = dict(cursor.fetchone())
                
                # Product distribution
                cursor.execute("""
                    SELECT product_name, COUNT(*) as count 
                    FROM assets 
                    WHERE product_name IS NOT NULL 
                    GROUP BY product_name
                """)
                product_distribution = dict(cursor.fetchall())
                
                return {
                    'total_records': total_records,
                    'state_distribution': state_distribution,
                    'voltage_stats': voltage_stats,
                    'product_distribution': product_distribution
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def clear_database(self) -> Dict[str, Any]:
        """Clear all data from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM assets")
                conn.commit()
                
                return {
                    'success': True,
                    'message': 'Database cleared successfully'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def clear_all_data(self):
        """Alias for clear_database to match interface."""
        return self.clear_database()