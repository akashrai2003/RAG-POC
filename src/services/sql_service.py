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
            'account_cus_id': 'Account_Cus_ID__c',
            'account_name': 'Account_Name__c',
            'account_to_update': 'Account_to_Update__c',
            'account_unique_id': 'Account_Unique_Id__c',
            'account_id': 'AccountId',
            'accuracy_meters': 'Accuracy_meters__c',
            'action_needed': 'Action_Needed__c',
            'action_needed_e1': 'Action_Needed_E1__c',
            'action_needed_from_e1': 'Action_Needed_From_E1__c',
            'altitude': 'Altitude__c',
            'asset_name': 'Asset_Name_formula__c',
            'asset_name_url': 'Asset_Name_URL__c',
            'asset_level': 'AssetLevel',
            'asset_provided_by': 'AssetProvidedById',
            'asset_serviced_by': 'AssetServicedById',
            'assigned_railway_contact_number': 'Assigned_Railway_Contact_Number__c',
            'assigned_railway_line': 'Assigned_Railway_Line__c',
            'async': 'async__c',
            'back_replaced_date': 'Back_Replaced_Date__c',
            'battery_check': 'Battery_check__c',
            'battery_replaced_date': 'Battery_Replaced_Date__c',
            'battery_voltage': 'Battery_Voltage__c',
            'business_location': 'Business_Location__c',
            'business_location_latitude': 'Business_Location__Latitude__s',
            'business_location_longitude': 'Business_Location__Longitude__s',
            'business_name': 'Business_Name__c',
            'business_name_link': 'Business_Name_link__c',
            'business_place_id': 'Business_Place_Id__c',
            'c_field': 'c__c',
            'capture_distance_records': 'Capture_Distance_Records__c',
            'capture_movement_event': 'Capture_Movement_Event__c',
            'capture_temperature_response': 'Capture_Temperature_Response__c',
            'capturing_response': 'Capturing_Response__c',
            'cardinal_tag': 'Cardinal_Tag__c',
            'checklane': 'CheckLane__c',
            'city': 'City',
            'config_change': 'Config_Change__c',
            'config_check_reported': 'Config_Check_Reported__c',
            'contact_id': 'ContactId',
            'country': 'Country',
            'created_by_id': 'CreatedById',
            'created_date': 'CreatedDate',
            'current_address': 'Current_Address__c',
            'current_city': 'Current_City__c',
            'current_country': 'Current_Country__c',
            'current_loc_assets_count': 'Current_Loc_Assets_count__c',
            'current_loc_latitude': 'Current_Loc_latitude__c',
            'current_loc_longitude': 'Current_Loc_Longitude__c',
            'current_location': 'Current_Location_Name__c',
            'current_location_address': 'Current_Location_Address__c',
            'current_location_report': 'Current_Location_Report__c',
            'current_location_unique_id': 'Current_Location_Unique_Id__c',
            'current_railway_line': 'Current_Railway_Line__c',
            'current_state': 'Current_State__c',
            'current_state2': 'Current_State2__c',
            'current_street_address': 'Current_Street_Address__c',
            'current_weight': 'Current_Weight__c',
            'current_zip_code': 'Current_Zip_Code__c',
            'custom_check_1': 'custom_check_1__c',
            'custom_check_2': 'custom_check_2__c',
            'customer_id_check': 'Customer_ID_Check__c',
            'date_shipped': 'Date_Shipped__c',
            'deleted_ahc': 'Deleted_AHC__c',
            'description': 'Description',
            'device_id': 'Device_Id__c',
            'digital_asset_status': 'DigitalAssetStatus',
            'distance': 'Distance__c',
            'distance_datetime': 'Distance_Datetime__c',
            'distance_level': 'Distance_Level__c',
            'dormant_days': 'Dormant_Days__c',
            'est_batterycalculate': 'est_Batterycalculate__c',
            'est_batt_pct': 'estBattPct__c',
            'event_name': 'eventName__c',
            'expected_delivery': 'Expected_Delivery__c',
            'facility_location': 'Facility_Location__c',
            'field1': 'Field1__c',
            'field2': 'Field2__c',
            'geocode_accuracy': 'GeocodeAccuracy',
            'humidity': 'Humidity__c',
            'iccid': 'ICCID__c',
            'id': 'Id',
            'imei': 'IMEI__c',
            'install_date': 'InstallDate',
            'is_access_point': 'Is_Access_Point__c',
            'is_asset_following_assigned_railway_line': 'Is_Asset_Following_Assigned_Railway_Line__c',
            'is_loco_asset': 'Is_Loco_Asset__c',
            'is_nimbe_link_asset': 'Is_NimbeLink_Asset__c',
            'ischanged_true': 'IschangedTrue__c',
            'is_competitor_product': 'IsCompetitorProduct',
            'is_deleted': 'IsDeleted',
            'is_internal': 'IsInternal',
            'is_new_asset': 'isNew_Asset__c',
            'label_replaced_date': 'Label_Replaced_Date__c',
            'last_connected': 'Last_Connected__c',
            'last_connected_map': 'Last_Connected_Map__c',
            'last_scan_date': 'Last_Scan_Date_ST__c',
            'last_synced_timestamp': 'Last_Synced_Timestamp__c',
            'latitude': 'Latitude',
            'longitude': 'Longitude',
            'manufacture_date': 'Manufacture_Date__c',
            'manufacturer': 'Manufacturer__c',
            'movement': 'Movement__c',
            'movement_end': 'Movement_End__c',
            'name': 'Name',
            'order_number': 'Order_Number__c',
            'order_status': 'Order_Status__c',
            'order_type': 'Order_Type__c',
            'owner_id': 'OwnerId',
            'pallet_type': 'Pallet_Type__c',
            'pause_job': 'Pause_Job__c',
            'plant_name': 'PlantName__c',
            'postal_code': 'PostalCode',
            'power_reset_occurred': 'Power_Reset_Occurred__c',
            'power_reset_time': 'Power_Reset_Time__c',
            'powerup_time': 'PowerUp_Time__c',
            'price': 'Price',
            'product_name': 'Product_Name__c',
            'product_code': 'Product_Code_2__c',
            'product_description': 'Product_Description__c',
            'quantity': 'Quantity',
            'rack_type': 'Rack_Type__c',
            'railway_contact_number': 'Railway_Contact_Number__c',
            'rssi': 'rssi__c',
            'serial_number': 'SerialNumber',
            'ship_to_customer_name': 'Ship_To_Customer_Name__c',
            'state_of_pallet': 'State_of_Pallet__c',
            'status': 'Status',
            'still_in_location_count': 'Still_in_Location_Count__c',
            'street': 'Street',
            'system_modstamp': 'SystemModstamp',
            'tag_destroyed_date': 'Tag_Destroyed_Date__c',
            'temperature': 'Temperature__c',
            'total_dwell_days': 'Total_Dwell_Days__c',
            'total_dwell_days_cl': 'Total_Dwell_Days_CL__c',
            'track_full_history': 'Track_Full_History__c',
            'account_address': 'Account_Address__c',
            'last_scan_date_alias': 'Date_Shipped__c'
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

        Schema Info : {self.field_mapping}
        
        User Request: {query_description}
        
        CRITICAL RULES for SAQL:
        
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
        
        6. **Best Practices:**
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
        SAQL: q = load "{self.dataset_id}/{self.dataset_version}"; q = foreach q generate q.Name as Asset, date_diff( \"day\", toDate(substr(Last_Connected__c, 1, 23) + \"Z\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\"), toDate(substr(Date_Shipped__c, 1, 23) + \"Z\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\")) as Days_Diff; q = filter q by Days_Diff < 30;"
        
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