# Wallet-Risk-Scoring-Model-Compound-V2-V3-Protocol

1. Data Collection Method
Implemented controlled synthetic data generation to create realistic Compound protocol transaction patterns with known ground truth labels for supervised learning.

Data Specifications:
Volume: 1,144 transactions across 103 unique wallet addresses
Time Range: 6-month historical window with realistic temporal distribution
Risk Distribution: 20% high-risk, 50% medium-risk, 30% low-risk wallets
Protocol Coverage: Compound V2 and V3 tokens (cUSDC, cETH, cDAI, cUSDT, cWBTC)

Transaction Attributes:
Actions: Supply, withdraw, borrow, repay operations
Amounts: Log-normal distribution ($50-$50,000) reflecting real DeFi usage patterns
Success Rates: Varied by risk profile (95% low-risk, 82% medium-risk, 65% high-risk)
Gas Patterns: Risk-correlated gas pricing behavior
Temporal Patterns: Realistic transaction frequency distributions



