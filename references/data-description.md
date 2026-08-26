# IEEE-CIS Fraud Detection — Data Description

Source: [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle competition).

## Objective

Predict the probability that an online transaction is fraudulent, as denoted by the
binary target `isFraud`.

## Data Structure

The data is split into **two tables**, joined on `TransactionID`:

| Table | Description |
|-------|-------------|
| `transaction` | Transaction-level information (amount, product, payment card, addresses, masked engineered features). |
| `identity` | Identity information — network connection info (IP, ISP, Proxy, etc.) and digital signature (UA / browser / OS / version, etc.), collected by Vesta's fraud protection system and digital security partners. |

> **Note:** Not all transactions have corresponding identity information
> (the `identity` table has fewer rows than `transaction`).

## Files

| File | Contents |
|------|----------|
| `train_transaction.csv` | Training set — transaction table. |
| `train_identity.csv` | Training set — identity table. |
| `test_transaction.csv` | Test set — transaction table (predict `isFraud` for these rows). |
| `test_identity.csv` | Test set — identity table. |
| `sample_submission.csv` | Sample submission file in the correct format. |

## Target Variable

- `isFraud` — binary label (`0` / `1`) indicating whether a transaction is fraudulent.
  Present only in the training data; predicted for the test data.

## Transaction Table — Features

| Feature group | Description |
|---------------|-------------|
| `TransactionDT` | Timedelta from a given reference datetime (**not** an actual timestamp). |
| `TransactionAMT` | Transaction payment amount in USD. |
| `ProductCD` | Product code — the product for each transaction. |
| `card1` – `card6` | Payment card information — card type, card category, issue bank, country, etc. |
| `addr1`, `addr2` | Address. |
| `dist` | Distance. |
| `P_emaildomain` / `R_emaildomain` | Purchaser and recipient email domain. |
| `C1` – `C14` | Counting features — e.g., how many addresses are found to be associated with the payment card, etc. The actual meaning is **masked**. |
| `D1` – `D15` | Timedelta features — e.g., days between previous transaction, etc. |
| `M1` – `M9` | Match features — e.g., names on card and address, etc. |
| `Vxxx` | Vesta engineered rich features — ranking, counting, and other entity relations. |

### Categorical Features — Transaction

- `ProductCD`
- `card1` – `card6`
- `addr1`, `addr2`
- `P_emaildomain`
- `R_emaildomain`
- `M1` – `M9`

## Identity Table — Features

Variables in this table are identity information — network connection information
(IP, ISP, Proxy, etc.) and digital signature (UA / browser / OS / version, etc.)
associated with transactions. They are collected by Vesta's fraud protection system
and digital security partners.

> **Note:** The field names are masked and the pairwise dictionary is **not provided**
> for privacy protection and contract agreement.

### Categorical Features — Identity

- `DeviceType`
- `DeviceInfo`
- `id_12` – `id_38`

## References

- [Competition overview / data discussion by the host](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203)
