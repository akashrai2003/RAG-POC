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
            'created date': 'CreatedDate',
            'date_shipped': 'Date_Shipped__c',
            'est_batterycalculate': 'est_Batterycalculate__c',
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
        """Generate SAQL query from natural language description."""
        prompt = f"""You are a SAQL expert. Generate ONLY the SAQL query string, no explanations or markdown.

Dataset: {self.dataset_id}/{self.dataset_version}
{schema_info}

Field Mappings:
battery/voltage→Battery_Voltage__c, state/status→State_of_Pallet__c, location→Current_Location_Name__c, product→Product_Name__c, account→Account_Name__c, asset_id→Asset_ID__c, last_connected→Last_Connected__c, date_shipped→Date_Shipped__c

WARNING: Last_Scan_Date_ST__c has inconsistent format (12-hour AM/PM) - AVOID using in date calculations

SAQL Rules:
- Structure: load→filter→foreach→limit (LIMIT AFTER FOREACH)
- String equality: ==, pattern: matches "*text*"
- Strings: DOUBLE quotes "value", Aliases: SINGLE quotes 'alias'
- Aggregations: count(), sum(field), avg(field), min(field), max(field)

Date Handling (CRITICAL):
- ONLY use: Last_Connected__c, Date_Shipped__c, CreatedDate (ISO format compatible)
- DO NOT use: Last_Scan_Date_ST__c (incompatible format)
- Pattern: substr(field, 1, 23) + "Z" then parse with "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
- Full: toDate(substr(field, 1, 23) + "Z", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
- date_diff("day", from_date, to_date) = to_date - from_date

Examples:

count all assets
q = load "{self.dataset_id}/{self.dataset_version}"; q = foreach q generate count() as 'Total';

assets with battery voltage less than 6
q = load "{self.dataset_id}/{self.dataset_version}"; q = filter q by Battery_Voltage__c < 6; q = foreach q generate Asset_ID__c, Battery_Voltage__c, Product_Name__c; q = limit q 100;

count assets by state
q = load "{self.dataset_id}/{self.dataset_version}"; q = group q by State_of_Pallet__c; q = foreach q generate State_of_Pallet__c, count() as 'Count';

assets not connected in last 7 days
q = load "{self.dataset_id}/{self.dataset_version}"; q = foreach q generate Asset_ID__c, Last_Connected__c, date_diff("day", toDate(substr(Last_Connected__c, 1, 23) + "Z", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"), now()) as 'Days_Since'; q = filter q by 'Days_Since' > 7; q = limit q 100;

assets where Date Shipped minus Last Connected is less than 30 days
q = load "{self.dataset_id}/{self.dataset_version}"; q = foreach q generate Asset_ID__c, Date_Shipped__c, Last_Connected__c, date_diff("day", toDate(substr(Last_Connected__c, 1, 23) + "Z", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"), toDate(substr(Date_Shipped__c, 1, 23) + "Z", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")) as 'Days_Diff'; q = filter q by 'Days_Diff' < 30; q = limit q 100;

days since shipment
q = load "{self.dataset_id}/{self.dataset_version}"; q = foreach q generate Asset_ID__c, Date_Shipped__c, date_diff("day", toDate(substr(Date_Shipped__c, 1, 23) + "Z", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"), now()) as 'Days_Since_Shipped'; q = limit q 100;

assets in state In Network
q = load "{self.dataset_id}/{self.dataset_version}"; q = filter q by State_of_Pallet__c == "In Network"; q = foreach q generate Asset_ID__c, State_of_Pallet__c; q = limit q 100;

Query: {query_description}
SAQL:"""
        
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
            print(f"Error generating SAQL query: {e}")
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
        """Get database schema information - concise version."""
        try:
            # Try to get dataset metadata from REST API (most reliable)
            access_token = self._get_access_token_cached()
            if access_token:
                metadata_url = f"{self.salesforce_url}/services/data/v53.0/wave/datasets/{self.dataset_id}"
                headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                }
                response = requests.get(metadata_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    dataset_info = response.json()
                    
                    # Try to get fields from dataset metadata
                    if 'fields' in dataset_info:
                        field_names = [field['name'] for field in dataset_info['fields']]
                        return "Available Fields: " + ", ".join(sorted(field_names))
                    
                    # Alternative: Get XMD metadata for fields
                    version_url = f"{self.salesforce_url}/services/data/v53.0/wave/datasets/{self.dataset_id}/versions/{self.dataset_version}"
                    version_response = requests.get(version_url, headers=headers, timeout=30)
                    
                    if version_response.status_code == 200:
                        version_data = version_response.json()
                        field_names = []
                        
                        if 'xmdMain' in version_data:
                            xmd = version_data['xmdMain']
                            if 'dimensions' in xmd:
                                field_names.extend([dim['field'] for dim in xmd['dimensions']])
                            if 'measures' in xmd:
                                field_names.extend([measure['field'] for measure in xmd['measures']])
                        
                        if field_names:
                            return "Available Fields: " + ", ".join(sorted(field_names))
            
            # Fallback: Return common field mappings we know about
            return "Available Fields: " + ", ".join(sorted(self.field_mapping.values()))
                
        except Exception as e:
            print(f"Schema retrieval error: {str(e)}")
            # Last resort: return known fields
            return "Available Fields: " + ", ".join(sorted(self.field_mapping.values()))
    
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