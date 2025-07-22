## Tasks

### 1, test Z vs E-N-Z
Try to traine the LSTM model the feature Type D (0-69)

For Z:
Input: Type D + EHZ, ILL12, 
Output: F1

For E-N-Z,
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3) # 1e-4 does not work for ENZ
```
Input: Type D + EHE-N-Z, ILL12, 
Output: F1