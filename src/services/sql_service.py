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
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import config, QueryConfig
from datetime import datetime

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
            api_key=config.openai_api_key,
            temperature=QueryConfig.AGENT_TEMPERATURE
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
            'account_address': 'Account_Address__c',
            'total_dwell_days': 'Total_Dwell_Days__c'
        }
        
        print(f"Einstein Analytics Service initialized")
        print(f"Instance: {self.salesforce_url}")
        print(f"Dataset: {self.dataset_id}/{self.dataset_version}")
    
    def _get_access_token_cached(self) -> Optional[str]:
        """Get cached token or fetch a new one."""
        if self._access_token is None:
            self._access_token = self._get_access_token()
        return self._access_token

    def _get_access_token(self, force_refresh: bool = False) -> str:
        """Authenticate with Salesforce and get access token."""
        # If forcing refresh, clear cached token
        if force_refresh:
            self._access_token = None
        
        # If we have a cached token and not forcing refresh, return it
        if self._access_token and not force_refresh:
            return self._access_token
        
        print(f"🔐 Authenticating with Salesforce...")
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
        print(f"✅ Successfully authenticated with Salesforce")
        return self._access_token
    
    def execute_natural_language_query(self, query_description: str, rag_context: Optional[str] = None) -> Dict[str, Any]:
        """Convert natural language query to SQL and execute."""
        try:
            print(f"\n{'='*80}")
            print(f"🔍 SQL SERVICE: Processing query")
            print(f"📝 Query: {query_description}")
            if rag_context:
                print(f"📚 RAG Context: Provided ({len(rag_context)} chars)")
            print(f"{'='*80}")
            
            # Get database schema
            schema_info = self._get_schema_info()
            print("Schema Info:", schema_info)  # Debug: show schema info
            # Generate SAQL query using LLM (with optional RAG context)
            print(f"🤖 Generating SAQL query using LLM...")
            saql_query = self._generate_saql_query(query_description, schema_info, rag_context)

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
    
    def _generate_saql_query(self, query_description: str, schema_info: str, rag_context: Optional[str] = None) -> Optional[str]:
        """Generate SAQL query from natural language description."""
        
        # Add RAG context section if provided
        rag_section = ""
        if rag_context:
            rag_section = f"""
BUSINESS CONTEXT FROM DOCUMENTATION:
{rag_context}

CRITICAL: Use this context to understand business logic and formulas.
- If the context defines HOW something is calculated (formulas, conditions), implement that logic in SAQL
- Don't just filter by status fields - implement the actual business rules
- Example: "Rogue Asset" might be calculated based on date differences, not just a status field

Map business terms to database columns AND their calculation logic.
"""
        
        prompt = f"""You are a SAQL expert which generates queries that are to be executed in Salesforce Einstein Analytics DB. Generate ONLY the SAQL query string, no explanations or markdown.
Today's date is {datetime.utcnow().strftime('%Y-%m-%d')}.
Dataset: {self.dataset_id}/{self.dataset_version}
{schema_info}

{rag_section}Field Mappings:
battery/voltage→Battery_Voltage__c, state/status→State_of_Pallet__c, location→Current_Location_Name__c, product→Product_Name__c, account→Account_Name__c, asset_id→Asset_ID__c, last_connected→Last_Connected__c, date_shipped→Date_Shipped__c, total_dwell_days/days_in_transit→Total_Dwell_Days__c

IMPORTANT: Total_Dwell_Days__c is a pre-calculated field that contains the number of days an asset has been in transit.
Use this field when queries ask about "rogue" assets, "days in transit", or time-based asset status.

User Request: {query_description}

IMPORTANT: If the Business Context above defines HOW to calculate something (like "Rogue Asset = days in transit >= limit"), 
implement that calculation logic in SAQL instead of just filtering by a status field.
        
        CRITICAL RULES for SAQL:
        - Ensure not to use SQL syntax, only SAQL. like 'load', 'filter', 'foreach', 'limit', 'date_add.
        1. **SAQL Syntax** (NOT SQL):
           - Always start with: q = load "{self.dataset_id}/{self.dataset_version}";
           - Use 'q = filter q by' for WHERE conditions
           - Use 'q = group q by' for GROUP BY operations
           - Use 'q = order q by' for ORDER BY
           - Use 'q = limit q' for limiting results
           - Use 'q = foreach q generate' for SELECT projections
        
        
        2. **User term mapping:**
           - "state" or "status" → State_of_Pallet__c
           - "location" → Current_Location_Name__c
           - "voltage" → Battery_Voltage__c
           - "product" → Product_Name__c
           - "account" → Account_Name__c
           - "id" → Asset_ID__c
        
        3. **SAQL Operators:**
           - String equality: == (not =)
           - String literals: Use DOUBLE quotes "value" (NOT single quotes 'value')
           - String pattern: matches "*pattern*"
           - Numerical: <, >, <=, >=, ==, !=
           - Logical: && (AND), || (OR), ! (NOT)
           - Boolean values: true, false (lowercase)
           - CRITICAL: String comparisons MUST use double quotes: State_of_Pallet__c == "In Network"
        
        4. **Aggregations:**
           - count() - counts ALL records (no field argument)
           - sum(field) - sum of numeric field
           - avg(field) - average of numeric field
           - min(field), max(field) - min/max of field
           - IMPORTANT: count() takes NO arguments, it counts rows
           - To count records, use: q = foreach q generate count() as 'Count';
           - To count with grouping: q = group q by Field__c; q = foreach q generate Field__c, count() as 'Count';
        
        5. **CRITICAL Query Statement Order:**
           - LIMIT must come AFTER FOREACH, never before
           - Correct order: load → filter → foreach → limit
           - When filtering data: ALWAYS use foreach before limit
           - Pattern: q = load → q = filter → q = foreach q generate → q = limit
        
        6. **Date Filtering Rules:**
           - NEVER use toDate() in pre-projection filters (before foreach)
           - NEVER use matches operator on DATE fields (only works on strings)
           - For date filters: MUST project date field first, THEN filter with date comparison operators
           - Pattern: load → foreach (with toDate conversion) → filter (on converted field using >=, <=) → limit
           - Example: q = foreach q generate toDate(substr(CreatedDate, 1, 23) + "Z", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'") as 'Created_Date', Asset_ID__c; q = filter q by 'Created_Date' >= toDate("2025-01-01T00:00:00.000Z", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'") && 'Created_Date' < toDate("2026-01-01T00:00:00.000Z", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"); q = limit q 100;
           - For year filtering: Use date range comparison (>= start of year && < start of next year)
           - CRITICAL: Use date comparison operators (>=, <=, <, >) NOT matches for dates
           - NEVER use date_trunc() - not supported in SAQL
           - For month/year grouping: Use date_to_string() to format dates, then group by the string
           - Example month grouping: q = foreach q generate date_to_string(toDate(CreatedDate), "yyyy-MM") as 'Month', Asset_ID__c; q = group q by 'Month'; q = foreach q generate 'Month', count() as 'Count';
        
        7. **Best Practices:**
           - While using toDate function in saql use the formatting substr(Field name, 1, 23) + \"Z\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\"
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
        
        Query: "assets with battery voltage less than 6"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = filter q by Battery_Voltage__c < 6; q = foreach q generate Asset_ID__c, Battery_Voltage__c, Product_Name__c, State_of_Pallet__c; q = limit q 100;
        
        Query: "count assets by state"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = group q by State_of_Pallet__c; q = foreach q generate State_of_Pallet__c, count() as 'Count';
        
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
        
        Query: "Sum of battery voltage for assets"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = foreach q generate 'Asset_ID__c', 'Battery_Voltage__c'; result = group q by 'Asset_ID__c'; result = foreach result generate sum(q.'Battery_Voltage__c') as total_voltage;

        Query: "Show the Assets where Last Scan Date - Last Connected is less than 30 days"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = foreach q generate q.Name as Asset, date_diff( \"day\", toDate(substr(Last_Connected__c, 1, 23) + \"Z\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\"), toDate(substr(Date_Shipped__c, 1, 23) + \"Z\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\")) as Days_Diff; q = filter q by Days_Diff < 30;
        
        Query: "Show all assets created in 2025"
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = foreach q generate Asset_ID__c, Account_Name__c, Product_Name__c, toDate(substr(CreatedDate, 1, 23) + \"Z\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\") as 'Created_Date'; q = filter q by 'Created_Date' >= toDate(\"2025-01-01T00:00:00.000Z\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\") && 'Created_Date' < toDate(\"2026-01-01T00:00:00.000Z\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\"); q = limit q 100;
        
        Query: "Which assets are rogue?" (when RAG context defines: Rogue = Total_Dwell_Days__c >= 14)
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = filter q by Total_Dwell_Days__c >= 14; q = foreach q generate Asset_ID__c, Product_Name__c, Account_Name__c, Current_Location_Name__c, Total_Dwell_Days__c, State_of_Pallet__c; q = limit q 100;
        
        Now generate SAQL for: {query_description}
        
        SAQL Query:"""
        
        try:
            response = self.llm.invoke(prompt)
            saql_query = response.content.strip()
            print("Raw LLM Response:", saql_query)  # Debug: show raw response
            # Clean up markdown formatting
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
            print(f"📊 Executing SAQL Query: {saql_query}")  # Debug: show actual query
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
            print("Response:", response)
            if response.status_code == 200:
                result = self._parse_saql_response(response.text)
                print("Parsed Result:", result)
                return {
                    'success': True,
                    'data': result
                }
            else:
                error = f"API Error {response.status_code}: {response.text}"
                print(error)

                # Handle 401 authentication errors with retry
                if response.status_code == 401:
                    print("⚠️  Session expired - refreshing token and retrying...")
                    # Force refresh the access token
                    access_token = self._get_access_token(force_refresh=True)
                    if access_token:
                        headers['Authorization'] = f'Bearer {access_token}'
                        print("🔄 Retrying query with new token...")
                        response = requests.post(request_url, headers=headers, json=request_body, timeout=30)
                        if response.status_code == 200:
                            result = self._parse_saql_response(response.text)
                            print("✅ Retry succeeded!")
                            return {
                                'success': True,
                                'data': result
                            }
                        else:
                            error = f"API Error after re-auth {response.status_code}: {response.text}"
                            print(f"❌ Retry failed: {error}")
                            return {
                                'success': False,
                                'error': error,
                                'data': []
                            }
                
                # Return error for non-401 failures
                return {
                    'success': False,
                    'error': error,
                    'data': []
                }
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