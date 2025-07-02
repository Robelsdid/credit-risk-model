# Placeholder for Pydantic models 
from pydantic import BaseModel
from typing import Optional

class CreditRiskRequest(BaseModel):
    TransactionId: Optional[float] = None
    BatchId: float
    AccountId: float
    SubscriptionId: float
    CurrencyCode: float
    CountryCode: float
    ProviderId: float
    ProductId: float
    ProductCategory: float
    ChannelId: float
    Amount: float
    Value: float
    PricingStrategy: float
    FraudResult: float
    Amount_sum: float
    Amount_mean: float
    Amount_count: float
    Amount_std: float
    Value_sum: float
    Value_mean: float
    Value_count: float
    Value_std: float
    transaction_hour: float
    transaction_day: float
    transaction_month: float
    transaction_year: float
    transaction_dayofweek: float
    transaction_quarter: float
    hour_sin: float
    hour_cos: float
    dayofweek_sin: float
    dayofweek_cos: float

class CreditRiskResponse(BaseModel):
    risk_probability: float 