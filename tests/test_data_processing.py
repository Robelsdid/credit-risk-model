import pandas as pd
import numpy as np
import pytest
from src.data_processing import AggregateFeatures, DateTimeFeatures

# Test AggregateFeatures
def test_aggregate_features():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [10, 20, 30],
        'Value': [100, 200, 300]
    })
    agg = AggregateFeatures()
    result = agg.fit_transform(df)
    # There should be new columns for aggregates
    for col in ['Amount_sum', 'Amount_mean', 'Amount_count', 'Amount_std',
                'Value_sum', 'Value_mean', 'Value_count', 'Value_std']:
        assert col in result.columns
    # Check correct aggregation for CustomerId 1
    row = result[result['CustomerId'] == 1].iloc[0]
    assert row['Amount_sum'] == 30
    assert row['Amount_mean'] == 15
    assert row['Amount_count'] == 2
    assert np.isclose(row['Amount_std'], 7.0710678, atol=1e-4)
    assert row['Value_sum'] == 300
    assert row['Value_mean'] == 150
    assert row['Value_count'] == 2
    assert np.isclose(row['Value_std'], 70.710678, atol=1e-3)

# Test DateTimeFeatures
def test_datetime_features():
    df = pd.DataFrame({
        'TransactionStartTime': ['2024-07-01 15:30:00', '2024-07-02 08:45:00']
    })
    dt = DateTimeFeatures()
    result = dt.fit_transform(df)
    # Check extracted features
    assert 'transaction_hour' in result.columns
    assert result.loc[0, 'transaction_hour'] == 15
    assert result.loc[1, 'transaction_hour'] == 8
    assert 'transaction_day' in result.columns
    assert result.loc[0, 'transaction_day'] == 1
    assert result.loc[1, 'transaction_day'] == 2
    assert 'hour_sin' in result.columns
    assert np.isclose(result.loc[0, 'hour_sin'], np.sin(2 * np.pi * 15 / 24))
    assert 'hour_cos' in result.columns
    assert np.isclose(result.loc[1, 'hour_cos'], np.cos(2 * np.pi * 8 / 24))
