"""
Einstein Analytics service for executing queries on asset data via REST API.
This service queries Salesforce Einstein Data Cloud directly - no data duplication needed.
"""

import os
import requests
import json
import re
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI

from config.settings import config, QueryConfig


class SQLService:
    """Service for querying Einstein Analytics via REST API."""
    
    def __init__(self):
        """Initialize Einstein Analytics service."""
        # Salesforce configuration from environment variables
        self.salesforce_url = os.getenv('SALESFORCE_INSTANCE_URL')
        self.salesforce_username = os.getenv('SALESFORCE_USERNAME')
        self.salesforce_password = os.getenv('SALESFORCE_PASSWORD')
        self.salesforce_client_id = os.getenv('SALESFORCE_CLIENT_ID')
        self.salesforce_client_secret = os.getenv('SALESFORCE_CLIENT_SECRET')
        self.dataset_id = os.getenv('EINSTEIN_DATASET_ID')
        self.dataset_version = os.getenv('EINSTEIN_DATASET_VERSION_ID', 'latest')
        
        # Cached access token
        self._access_token = None
        
        # Initialize LLM for SAQL generation
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=config.openai_api_key,
            temperature=QueryConfig.SQL_GENERATION_TEMPERATURE
        )
        
        # Field mapping: friendly names -> Salesforce names
        self.field_mapping = {
            'asset_id': 'Asset_ID__c',
            'account_name': 'Account_Name__c',
            'action_needed': 'Action_Needed__c',
            'battery_voltage': 'Battery_Voltage__c',
            'cardinal_tag': 'Cardinal_Tag__c',
            'current_location': 'Current_Location_Name__c',
            'date_shipped': 'Date_Shipped__c',
            'est_battery_calculate': 'Est_Battery_Calculate__c',
            'last_connected': 'Last_Connected__c',
            'power_reset_occurred': 'Power_Reset_Occurred__c',
            'power_reset_time': 'Power_Reset_Time__c',
            'powerup_time': 'Powerup_Time__c',
            'product_name': 'Product_Name__c',
            'state_of_pallet': 'State_of_Pallet__c',
            'account_address': 'Account_Address__c'
        }
        
        print(f"Einstein Analytics Service initialized")
        print(f"Instance: {self.salesforce_url}")
        print(f"Dataset: {self.dataset_id}/{self.dataset_version}")
    
    def _get_access_token_cached(self) -> Optional[str]:
        if self._access_token is None:
            self._access_token = self._get_access_token()
        return self._access_token

    def _get_access_token(self) -> str:
        """Authenticate with Salesforce and get access token."""
        if self._access_token:
            return self._access_token
        
        auth_url = "https://test.salesforce.com/services/oauth2/token"
        payload = {
            'grant_type': 'password',
            'client_id': self.salesforce_client_id,
            'client_secret': self.salesforce_client_secret,
            'username': self.salesforce_username,
            'password': self.salesforce_password
        }
        
        response = requests.post(auth_url, data=payload)
        response.raise_for_status()
        
        auth_data = response.json()
        self._access_token = auth_data['access_token']
        return self._access_token
    
    def execute_natural_language_query(self, query_description: str) -> Dict[str, Any]:
        """Convert natural language query to SQL and execute."""
        try:
            print(f"\n{'='*80}")
            print(f"🔍 SQL SERVICE: Processing query")
            print(f"📝 Query: {query_description}")
            print(f"{'='*80}")
            
            # Get database schema
            schema_info = self._get_schema_info()
            
            # Generate SAQL query using LLM
            print(f"🤖 Generating SAQL query using OpenAI...")
            saql_query = self._generate_saql_query(query_description, schema_info)

            if not saql_query:
                print(f"❌ Failed to generate SAQL query")
                return {
                    'success': False,
                    'error': 'Could not generate SAQL query'
                }
            
            print(f"✅ Generated SAQL Query:")
            print(f"   {saql_query}")
            
            # Execute the query
            print(f"⚡ Executing SAQL query...")
            result = self._query_saql(saql_query)
            result['saql_query'] = saql_query
            
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
    
    def _generate_saql_query(self, query_description: str, schema_info: str) -> Optional[str]:
        """Generate SQL query from natural language description."""
        prompt = f"""
        You are an expert SAQL (Salesforce Analytics Query Language) generator for Einstein Data Cloud.
        
        Dataset Information:
        {schema_info}
        
        User Request: {query_description}
        
        CRITICAL RULES for SAQL:
        
        1. **SAQL Syntax** (NOT SQL):
           - Always start with: q = load "{self.dataset_id}/{self.dataset_version}";
           - Use 'q = filter q by' for WHERE conditions
           - Use 'q = group q by' for GROUP BY operations
           - Use 'q = order q by' for ORDER BY
           - Use 'q = limit q' for limiting results
           - Use 'q = foreach q generate' for SELECT projections
        
        2. **Exact Salesforce Field Names** (case-sensitive with __c suffix):
           - Asset_ID__c (NOT asset_id)
           - Account_Name__c (NOT account_name)
           - Action_Needed__c
           - Battery_Voltage__c (NOT battery_voltage)
           - Cardinal_Tag__c
           - Current_Location_Name__c (NOT current_location)
           - Date_Shipped__c
           - Est_Battery_Calculate__c
           - Last_Connected__c
           - Power_Reset_Occurred__c
           - Power_Reset_Time__c
           - Powerup_Time__c
           - Product_Name__c (NOT product_name)
           - State_of_Pallet__c (NOT state_of_pallet)
           - Account_Address__c
        
        3. **User term mapping:**
           - "state" or "status" → State_of_Pallet__c
           - "location" → Current_Location_Name__c
           - "voltage" → Battery_Voltage__c
           - "product" → Product_Name__c
           - "account" → Account_Name__c
           - "id" → Asset_ID__c
        
        4. **SAQL Operators:**
           - String equality: == (not =)
           - String literals: Use DOUBLE quotes "value" (NOT single quotes 'value')
           - String pattern: matches "*pattern*"
           - Numerical: <, >, <=, >=, ==, !=
           - Logical: && (AND), || (OR), ! (NOT)
           - Boolean values: true, false (lowercase)
           - CRITICAL: String comparisons MUST use double quotes: State_of_Pallet__c == "In Network"
        
        5. **Aggregations:**
           - count() - counts ALL records (no field argument)
           - sum(field) - sum of numeric field
           - avg(field) - average of numeric field
           - min(field), max(field) - min/max of field
           - IMPORTANT: count() takes NO arguments, it counts rows
           - To count records, use: q = foreach q generate count() as 'Count';
           - To count with grouping: q = group q by Field__c; q = foreach q generate Field__c, count() as 'Count';
        
        6. **CRITICAL Query Statement Order:**
           - LIMIT must come AFTER FOREACH, never before
           - Correct order: load → filter → foreach → limit
           - When filtering data: ALWAYS use foreach before limit
           - Pattern: q = load → q = filter → q = foreach q generate → q = limit
        
        7. **Best Practices:**
           - Always add limit at the END after foreach (limit 100 unless user specifies different)
           - Use single quotes for string literals
           - For filtered queries, MUST use foreach before limit
           - When selecting all fields, use 'q = foreach q generate' with field names or use load → limit (no filter)
           - Use DOUBLE quotes for string literals: "In Network", "Active", etc.
           - Use single quotes ONLY for aliases: count() as 'Total_Count'
        Return ONLY the SAQL query string, no explanations, no markdown, no JSON.
        
        Examples:
        
        Query: "count all assets"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = foreach q generate count() as 'Total_Count';
        
        Query: "count of Asset_ID__c"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = foreach q generate count() as 'Total_Count';
        
        Query: "assets with battery voltage less than 6"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = filter q by Battery_Voltage__c < 6; q = foreach q generate Asset_ID__c, Battery_Voltage__c, Product_Name__c, State_of_Pallet__c; q = limit q 100;
        
        Query: "count assets by state"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = group q by State_of_Pallet__c; q = foreach q generate State_of_Pallet__c, count() as 'Count';
        
        Query: "count by product"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = group q by Product_Name__c; q = foreach q generate Product_Name__c, count() as 'Count';
        
        Query: "show all assets"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = limit q 100;
        
        Query: "average voltage by product"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = group q by Product_Name__c; q = foreach q generate Product_Name__c, avg(Battery_Voltage__c) as 'Avg_Voltage';
        
        Query: "assets not connected in last 7 days"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = foreach q generate Asset_ID__c, Last_Connected__c, date_diff("day", toDate(Last_Connected__c), now()) as 'Days_Since_Connected'; q = filter q by 'Days_Since_Connected' > 7; q = limit q 100;
        
        Query: "days since shipment for each asset"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = foreach q generate Asset_ID__c, Product_Name__c, Date_Shipped__c, date_diff("day", toDate(Date_Shipped__c), now()) as 'Days_Since_Shipped'; q = limit q 100;
        
        Query: "assets with low voltage in California"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = filter q by Battery_Voltage__c < 6 && Current_Location_Name__c matches "*California*"; q = foreach q generate Asset_ID__c, Battery_Voltage__c, Current_Location_Name__c; q = limit q 100;
        
        Query: "assets with state In Network"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = filter q by State_of_Pallet__c == "In Network"; q = foreach q generate Asset_ID__c, State_of_Pallet__c, Product_Name__c; q = limit q 100;
        
        Query: "count assets in state In Network"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = filter q by State_of_Pallet__c == "In Network"; q = foreach q generate count() as 'Count';
        
        Query: "assets with product AT3 Pilot"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = filter q by Product_Name__c == "AT3 Pilot"; q = foreach q generate Asset_ID__c, Product_Name__c, Battery_Voltage__c; q = limit q 100;
        
        Now generate SAQL for: {query_description}
        
        SAQL Query:
        """
        
        try:
            response = self.llm.invoke(prompt)
            saql_query = response.content.strip()
            
            # Clean up the response (remove any markdown formatting)
            saql_query = re.sub(r'```saql\n?', '', saql_query)
            saql_query = re.sub(r'```\n?', '', saql_query)
            saql_query = saql_query.strip()
            
            if "load" not in saql_query.lower():
                return None

            return saql_query
            
        except Exception as e:
            print(f"Error generating SQL query: {e}")
            return None

    def _query_saql(self, saql_query: str) -> Dict[str, Any]:
        """Execute SQL query and return results."""
        try:
            access_token = self._get_access_token_cached()
            if not access_token:
                return {
                    'success': False,
                    'error': 'Could not obtain access token',
                    'data': []
                }
            request_url = f"{self.salesforce_url}/services/data/v53.0/wave/query"
            request_body = {"query": saql_query}
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }

            response = requests.post(request_url, headers=headers, json=request_body, timeout=30)

            if response.status_code == 200:
                result = self._parse_saql_response(response.text)
                return {
                    'success': True,
                    'data': result
                }
            else:
                error = f"API Error {response.status_code}: {response.text}"
                print(error)

                if response.status_code == 401:
                    self._access_token = None  # Clear cached token
                    access_token = self._get_access_token_cached()
                    if access_token:
                        headers['Authorization'] = f'Bearer {access_token}'
                        response = requests.post(request_url, headers=headers, json=request_body, timeout=30)
                        if response.status_code == 200:
                            result = self._parse_saql_response(response.text)
                            return {
                                'success': True,
                                'data': result
                            }
                        else:
                            error = f"API Error after re-auth {response.status_code}: {response.text}"
                            print(error)
        except Exception as e:
            error = f"Exception during SAQL query: {str(e)}"
            print(error)
            return {
                "success": False,
                "error": error,
                "data": []
            }

    def _parse_saql_response(self, response_text: str) -> List[Dict[str, Any]]:
        try:
            response_data = json.loads(response_text)
            if 'results' in response_data:
                results = response_data['results']
                if 'records' in results:
                    return results.get('records', [])
        except Exception as e:
            print(f"Error parsing SAQL response: {str(e)}")
            return []
            

    def _get_schema_info(self) -> str:
        """Get database schema information."""
        try:
            schema_info += f"\nDataset ID: {self.dataset_id}\n"
            sample_saql = f'q = load "{self.dataset_id}/{self.dataset_version}"; q = limit q 3;'
            results = self._query_saql(sample_saql)
            if results and isinstance(results, list) and len(results) > 0:
                sample_record = results[0]
                schema_info += "Sample Record Fields:\n"
                for field in sample_record.keys():
                    schema_info += f"- {field}\n"
                return schema_info
                
        except Exception as e:
            return f"Could not retrieve schema: {str(e)}"
    
    # def get_database_stats(self) -> Dict[str, Any]:
    #     """Get database statistics."""
    #     try:
    #         with sqlite3.connect(self.db_path) as conn:
    #             cursor = conn.cursor()
                
    #             # Total records
    #             cursor.execute("SELECT COUNT(*) FROM assets")
    #             total_records = cursor.fetchone()[0]
                
    #             # Records by state
    #             cursor.execute("""
    #                 SELECT state_of_pallet, COUNT(*) as count 
    #                 FROM assets 
    #                 WHERE state_of_pallet IS NOT NULL 
    #                 GROUP BY state_of_pallet
    #             """)
    #             state_distribution = dict(cursor.fetchall())
                
    #             # Battery voltage stats
    #             cursor.execute("""
    #                 SELECT 
    #                     COUNT(*) as count,
    #                     AVG(battery_voltage) as avg_voltage,
    #                     MIN(battery_voltage) as min_voltage,
    #                     MAX(battery_voltage) as max_voltage
    #                 FROM assets 
    #                 WHERE battery_voltage IS NOT NULL
    #             """)
    #             voltage_stats = dict(cursor.fetchone())
                
    #             # Product distribution
    #             cursor.execute("""
    #                 SELECT product_name, COUNT(*) as count 
    #                 FROM assets 
    #                 WHERE product_name IS NOT NULL 
    #                 GROUP BY product_name
    #             """)
    #             product_distribution = dict(cursor.fetchall())
                
    #             return {
    #                 'total_records': total_records,
    #                 'state_distribution': state_distribution,
    #                 'voltage_stats': voltage_stats,
    #                 'product_distribution': product_distribution
    #             }
                
    #     except Exception as e:
    #         return {'error': str(e)}