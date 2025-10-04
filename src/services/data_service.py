"""
Data processing service for handling file uploads and data validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, BinaryIO
import time
from datetime import datetime
import io

from models.schemas import AssetDataFrame, AssetRecord, UploadResult
from services.sql_service import SQLService
from services.rag_service import RAGService


class DataProcessingService:
    """Service for processing uploaded asset data files."""
    
    def __init__(self, sql_service: SQLService, rag_service: RAGService):
        self.sql_service = sql_service
        self.rag_service = rag_service
    
    def process_uploaded_file(self, file_content: BinaryIO, filename: str) -> UploadResult:
        """Process uploaded Excel/CSV file and store in databases."""
        start_time = time.time()
        
        try:
            # Read the file based on extension
            df = self._read_file(file_content, filename)
            
            if df is None or df.empty:
                return UploadResult(
                    success=False,
                    filename=filename,
                    records_processed=0,
                    errors=["Could not read file or file is empty"],
                    processing_time=time.time() - start_time
                )
            
            # Clean and validate the data
            cleaned_df, cleaning_errors = self._clean_dataframe(df)
            
            # Convert to asset records
            try:
                asset_df = AssetDataFrame(cleaned_df)
                asset_records = asset_df.to_records()
            except Exception as e:
                return UploadResult(
                    success=False,
                    filename=filename,
                    records_processed=0,
                    errors=[f"Data validation failed: {str(e)}"],
                    processing_time=time.time() - start_time
                )
            
            if not asset_records:
                return UploadResult(
                    success=False,
                    filename=filename,
                    records_processed=0,
                    errors=["No valid asset records found in file"],
                    processing_time=time.time() - start_time
                )
            
            # Store in SQL database
            sql_result = self.sql_service.insert_assets(asset_records)
            
            # Store in vector database
            rag_result = self.rag_service.add_asset_documents(asset_records)
            
            # Combine results
            total_errors = cleaning_errors.copy()
            if not sql_result.get('success'):
                total_errors.append(f"SQL storage failed: {sql_result.get('error')}")
            if sql_result.get('errors'):
                total_errors.extend(sql_result['errors'])
            
            if not rag_result.get('success'):
                total_errors.append(f"RAG storage failed: {rag_result.get('error')}")
            
            # Get schema validation info
            schema_validation = self._validate_schema(cleaned_df)
            
            processing_time = time.time() - start_time
            
            return UploadResult(
                success=len(total_errors) == 0,
                filename=filename,
                records_processed=len(asset_records),
                errors=total_errors,
                schema_validation=schema_validation,
                processing_time=processing_time
            )
            
        except Exception as e:
            return UploadResult(
                success=False,
                filename=filename,
                records_processed=0,
                errors=[f"Processing failed: {str(e)}"],
                processing_time=time.time() - start_time
            )
    
    def _read_file(self, file_content: BinaryIO, filename: str) -> Optional[pd.DataFrame]:
        """Read file content based on file extension."""
        try:
            file_extension = filename.lower().split('.')[-1]
            
            if file_extension in ['xlsx', 'xls']:
                return pd.read_excel(file_content)
            elif file_extension == 'csv':
                return pd.read_csv(file_content)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            return None
    
    def _clean_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """Clean and standardize the dataframe."""
        errors = []
        
        try:
            # Make a copy to avoid modifying original
            cleaned_df = df.copy()
            
            # Remove completely empty rows
            cleaned_df = cleaned_df.dropna(how='all')
            
            # Handle missing Asset_ID__c
            if 'Asset_ID__c' not in cleaned_df.columns:
                errors.append("Required column 'Asset_ID__c' not found")
                return cleaned_df, errors
            
            # Remove rows without Asset_ID
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.dropna(subset=['Asset_ID__c'])
            removed_count = initial_count - len(cleaned_df)
            if removed_count > 0:
                errors.append(f"Removed {removed_count} rows with missing Asset_ID")
            
            # Clean numeric columns
            numeric_columns = ['Battery_Voltage__c', 'est_Batterycalculate__c', 'PowerUp_Time__c']
            for col in numeric_columns:
                if col in cleaned_df.columns:
                    # Replace 'NaN' strings with actual NaN
                    cleaned_df[col] = cleaned_df[col].replace(['NaN', 'nan', '', 'null'], np.nan)
                    # Convert to numeric, errors='coerce' will turn invalid values to NaN
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            
            # Clean date columns
            date_columns = ['Date_Shipped__c', 'Last_Connected__c', 'Power_Reset_Time__c']
            for col in date_columns:
                if col in cleaned_df.columns:
                    # Replace 'NaN' strings with actual NaN
                    cleaned_df[col] = cleaned_df[col].replace(['NaN', 'nan', '', 'null'], np.nan)
                    # Convert to datetime
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
            
            # Clean boolean columns
            boolean_columns = ['Cardinal_Tag__c', 'Power_Reset_Occurred__c']
            for col in boolean_columns:
                if col in cleaned_df.columns:
                    # Replace 'NaN' strings with actual NaN
                    cleaned_df[col] = cleaned_df[col].replace(['NaN', 'nan', '', 'null'], np.nan)
                    # Convert boolean-like values
                    cleaned_df[col] = cleaned_df[col].map({
                        True: True, False: False,
                        'true': True, 'false': False,
                        'True': True, 'False': False,
                        1: True, 0: False,
                        '1': True, '0': False,
                        'yes': True, 'no': False,
                        'Yes': True, 'No': False
                    })
            
            # Clean string columns - remove extra whitespace
            string_columns = ['Account_Name__c', 'Action_Needed__c', 'Current_Location_Name__c', 
                            'Product_Name__c', 'State_of_Pallet__c', 'Account.Address__c']
            for col in string_columns:
                if col in cleaned_df.columns:
                    # Replace 'NaN' strings with actual NaN
                    cleaned_df[col] = cleaned_df[col].replace(['NaN', 'nan', 'null'], np.nan)
                    # Strip whitespace from string values
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                    # Replace empty strings with NaN
                    cleaned_df[col] = cleaned_df[col].replace('', np.nan)
            
            # Validate data ranges
            if 'Battery_Voltage__c' in cleaned_df.columns:
                # Check for unrealistic battery voltage values
                voltage_outliers = cleaned_df[
                    (cleaned_df['Battery_Voltage__c'] < 0) | 
                    (cleaned_df['Battery_Voltage__c'] > 20)
                ]['Asset_ID__c'].tolist()
                
                if voltage_outliers:
                    errors.append(f"Found unrealistic battery voltage values for assets: {voltage_outliers[:5]}")
            
            return cleaned_df, errors
            
        except Exception as e:
            errors.append(f"Data cleaning failed: {str(e)}")
            return df, errors
    
    def _validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the dataframe schema and provide feedback."""
        validation_info = {
            'total_columns': len(df.columns),
            'total_rows': len(df),
            'required_columns_present': [],
            'optional_columns_present': [],
            'missing_columns': [],
            'data_quality': {}
        }
        
        # Define expected columns
        required_columns = ['Asset_ID__c']
        optional_columns = [
            'Account_Name__c', 'Action_Needed__c', 'Battery_Voltage__c',
            'Cardinal_Tag__c', 'Current_Location_Name__c', 'Date_Shipped__c',
            'est_Batterycalculate__c', 'Last_Connected__c', 'Power_Reset_Occurred__c',
            'Power_Reset_Time__c', 'PowerUp_Time__c', 'Product_Name__c',
            'State_of_Pallet__c', 'Account.Address__c'
        ]
        
        # Check which columns are present
        for col in required_columns:
            if col in df.columns:
                validation_info['required_columns_present'].append(col)
            else:
                validation_info['missing_columns'].append(col)
        
        for col in optional_columns:
            if col in df.columns:
                validation_info['optional_columns_present'].append(col)
        
        # Data quality checks
        for col in df.columns:
            if col in required_columns + optional_columns:
                null_count = df[col].isnull().sum()
                null_percentage = (null_count / len(df)) * 100
                
                validation_info['data_quality'][col] = {
                    'null_count': int(null_count),
                    'null_percentage': round(null_percentage, 2),
                    'unique_values': int(df[col].nunique()),
                    'data_type': str(df[col].dtype)
                }
        
        return validation_info
    
    def create_sample_data(self) -> Dict[str, Any]:
        """Create sample asset data for testing."""
        sample_data = [
            {
                "Asset_ID__c": "at-atp3cb2aa3080a",
                "Account_Name__c": "Cardinal IG Spring Green",
                "Action_Needed__c": "RMA to SMART",
                "Battery_Voltage__c": 4.14,
                "Cardinal_Tag__c": True,
                "Current_Location_Name__c": "Keene's Transfer, Inc.",
                "Date_Shipped__c": "2025-05-30T03:30:42.000+0530",
                "est_Batterycalculate__c": 0.0,
                "Last_Connected__c": "2024-08-29T08:15:20.000+0530",
                "Power_Reset_Occurred__c": True,
                "Power_Reset_Time__c": "2021-11-05T22:29:20.000+0530",
                "PowerUp_Time__c": 12222575,
                "Product_Name__c": "AT3 Pilot",
                "State_of_Pallet__c": "In Network",
                "Account.Address__c": "1011 East Madison Street, Spring Green, WI 53588, US"
            },
            {
                "Asset_ID__c": "at-atp302bfa30892",
                "Account_Name__c": "Deactivated Tags - Returned/retired AT3s",
                "Action_Needed__c": None,
                "Battery_Voltage__c": 4.57,
                "Cardinal_Tag__c": True,
                "Current_Location_Name__c": None,
                "Date_Shipped__c": "2025-03-03T23:13:37.000+0530",
                "est_Batterycalculate__c": 0.0,
                "Last_Connected__c": "2025-03-20T04:22:31.000+0530",
                "Power_Reset_Occurred__c": False,
                "Power_Reset_Time__c": None,
                "PowerUp_Time__c": 171267625,
                "Product_Name__c": "AT3 Pilot",
                "State_of_Pallet__c": "In Transit",
                "Account.Address__c": None
            },
            {
                "Asset_ID__c": "at5-s4ef37a8dd9d",
                "Account_Name__c": "Returned for Refurbishment - DEACTIVATED",
                "Action_Needed__c": "RMA to SMART",
                "Battery_Voltage__c": 5.05,
                "Cardinal_Tag__c": True,
                "Current_Location_Name__c": None,
                "Date_Shipped__c": "2024-11-12T05:46:59.000+0530",
                "est_Batterycalculate__c": 66.5,
                "Last_Connected__c": "2024-06-26T09:37:43.000+0530",
                "Power_Reset_Occurred__c": False,
                "Power_Reset_Time__c": None,
                "PowerUp_Time__c": 83873725,
                "Product_Name__c": "AT5 - Bracket",
                "State_of_Pallet__c": "In Transit",
                "Account.Address__c": None
            },
            {
                "Asset_ID__c": "at5-s33ef0af285a",
                "Account_Name__c": "Deactivated Tags - Not Returned",
                "Action_Needed__c": "RMA to SMART",
                "Battery_Voltage__c": 5.82,
                "Cardinal_Tag__c": True,
                "Current_Location_Name__c": "Cardinal IG - Fremont IN",
                "Date_Shipped__c": "2024-11-07T19:28:41.000+0530",
                "est_Batterycalculate__c": 65.6,
                "Last_Connected__c": "2024-02-03T08:40:37.000+0530",
                "Power_Reset_Occurred__c": False,
                "Power_Reset_Time__c": None,
                "PowerUp_Time__c": 71426525,
                "Product_Name__c": "AT5 - Bracket",
                "State_of_Pallet__c": "In Network",
                "Account.Address__c": None
            },
            {
                "Asset_ID__c": "at5-s34515b73b6a",
                "Account_Name__c": "Cardinal IG Fargo",
                "Action_Needed__c": None,
                "Battery_Voltage__c": 6.3,
                "Cardinal_Tag__c": True,
                "Current_Location_Name__c": "Cardinal IG Fargo",
                "Date_Shipped__c": None,
                "est_Batterycalculate__c": 75.5,
                "Last_Connected__c": "2025-09-29T07:16:25.000+0530",
                "Power_Reset_Occurred__c": False,
                "Power_Reset_Time__c": None,
                "PowerUp_Time__c": 123609025,
                "Product_Name__c": "AT5 - Bracket",
                "State_of_Pallet__c": "In Network",
                "Account.Address__c": "4611 15th Avenue N W, Fargo, ND 58102, US"
            }
        ]
        
        # Convert to DataFrame and process
        df = pd.DataFrame(sample_data)
        
        # Create a file-like object
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        # Process as if it were an uploaded file
        return self.process_uploaded_file(
            io.BytesIO(csv_buffer.getvalue().encode()), 
            "sample_data.csv"
        )