"""
Data models and schemas for the RAG-POC application.
"""

from datetime import datetime
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field, validator
import pandas as pd


class AssetRecord(BaseModel):
    """Model for individual asset records."""
    
    asset_id: str = Field(..., description="Unique asset identifier")
    account_name: Optional[str] = Field(None, description="Account/customer name")
    action_needed: Optional[str] = Field(None, description="Required action")
    battery_voltage: Optional[float] = Field(None, description="Battery voltage reading")
    cardinal_tag: Optional[bool] = Field(None, description="Cardinal tag status")
    current_location: Optional[str] = Field(None, description="Current location")
    date_shipped: Optional[datetime] = Field(None, description="Shipping date")
    est_battery_calculate: Optional[float] = Field(None, description="Estimated battery calculation")
    last_connected: Optional[datetime] = Field(None, description="Last connection timestamp")
    power_reset_occurred: Optional[bool] = Field(None, description="Power reset status")
    power_reset_time: Optional[datetime] = Field(None, description="Power reset timestamp")
    powerup_time: Optional[int] = Field(None, description="Power up time in seconds")
    product_name: Optional[str] = Field(None, description="Product name/type")
    state_of_pallet: Optional[str] = Field(None, description="Current pallet state")
    account_address: Optional[str] = Field(None, description="Account address")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class QueryRequest(BaseModel):
    """Model for user query requests."""
    
    query: str = Field(..., min_length=1, max_length=500, description="User query")
    query_type: Optional[str] = Field(None, description="Detected query type (sql/rag)")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class QueryResponse(BaseModel):
    """Model for query responses."""
    
    success: bool = Field(..., description="Whether query was successful")
    query_type: str = Field(..., description="Type of query processed (sql/rag)")
    response: str = Field(..., description="Generated response text")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Structured data results")
    sql_query: Optional[str] = Field(None, description="Generated SQL query if applicable")
    execution_time: float = Field(..., description="Query execution time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class UploadResult(BaseModel):
    """Model for file upload results."""
    
    success: bool = Field(..., description="Whether upload was successful")
    filename: str = Field(..., description="Name of uploaded file")
    records_processed: int = Field(..., description="Number of records processed")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    schema_validation: Dict[str, Any] = Field(default_factory=dict, description="Schema validation results")
    processing_time: float = Field(..., description="Processing time in seconds")


class DatabaseStats(BaseModel):
    """Model for database statistics."""
    
    total_records: int = Field(..., description="Total number of records")
    last_updated: datetime = Field(..., description="Last database update timestamp")
    table_info: Dict[str, Any] = Field(default_factory=dict, description="Table structure information")
    vector_store_size: Optional[int] = Field(None, description="Number of vectors in ChromaDB")


class AssetDataFrame:
    """Wrapper class for asset data operations using pandas."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._validate_dataframe()
    
    def _validate_dataframe(self):
        """Validate the dataframe structure."""
        required_columns = ['Asset_ID__c']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def to_records(self) -> List[AssetRecord]:
        """Convert dataframe to list of AssetRecord objects."""
        records = []
        
        for _, row in self.df.iterrows():
            try:
                # Clean and convert the data
                record_data = {}
                
                # Map DataFrame columns to AssetRecord fields
                column_mapping = {
                    'Asset_ID__c': 'asset_id',
                    'Account_Name__c': 'account_name',
                    'Action_Needed__c': 'action_needed',
                    'Battery_Voltage__c': 'battery_voltage',
                    'Cardinal_Tag__c': 'cardinal_tag',
                    'Current_Location_Name__c': 'current_location',
                    'Date_Shipped__c': 'date_shipped',
                    'est_Batterycalculate__c': 'est_battery_calculate',
                    'Last_Connected__c': 'last_connected',
                    'Power_Reset_Occurred__c': 'power_reset_occurred',
                    'Power_Reset_Time__c': 'power_reset_time',
                    'PowerUp_Time__c': 'powerup_time',
                    'Product_Name__c': 'product_name',
                    'State_of_Pallet__c': 'state_of_pallet',
                    'Account.Address__c': 'account_address'
                }
                
                for df_col, model_field in column_mapping.items():
                    if df_col in row.index and pd.notna(row[df_col]):
                        value = row[df_col]
                        
                        # Handle datetime conversions
                        if model_field in ['date_shipped', 'last_connected', 'power_reset_time']:
                            if isinstance(value, str):
                                try:
                                    value = pd.to_datetime(value).to_pydatetime()
                                except:
                                    value = None
                            elif isinstance(value, pd.Timestamp):
                                value = value.to_pydatetime()
                        
                        # Handle boolean conversions
                        elif model_field in ['cardinal_tag', 'power_reset_occurred']:
                            if isinstance(value, str):
                                value = value.lower() in ['true', '1', 'yes']
                            else:
                                value = bool(value)
                        
                        # Handle numeric conversions
                        elif model_field in ['battery_voltage', 'est_battery_calculate']:
                            try:
                                value = float(value)
                            except:
                                value = None
                        
                        elif model_field == 'powerup_time':
                            try:
                                value = int(value)
                            except:
                                value = None
                        
                        record_data[model_field] = value
                
                if 'asset_id' in record_data:  # Only create record if we have the required field
                    records.append(AssetRecord(**record_data))
                    
            except Exception as e:
                # Log the error but continue processing other records
                print(f"Error processing row: {e}")
                continue
        
        return records
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset."""
        stats = {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'numeric_columns': list(self.df.select_dtypes(include=['number']).columns),
            'date_columns': list(self.df.select_dtypes(include=['datetime']).columns),
            'missing_data': self.df.isnull().sum().to_dict()
        }
        
        # Add specific asset statistics
        if 'Battery_Voltage__c' in self.df.columns:
            voltage_col = self.df['Battery_Voltage__c'].dropna()
            if len(voltage_col) > 0:
                stats['battery_voltage_stats'] = {
                    'mean': float(voltage_col.mean()),
                    'min': float(voltage_col.min()),
                    'max': float(voltage_col.max()),
                    'count': int(voltage_col.count())
                }
        
        if 'State_of_Pallet__c' in self.df.columns:
            stats['state_distribution'] = self.df['State_of_Pallet__c'].value_counts().to_dict()
        
        return stats