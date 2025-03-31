# LLM Test Results

## Model: deepseek-r1-1.5B-128k, File Extension: csv, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 1.5 | 1.7 | 1.7 | 0.454 | 0.361 | 0.315 | 0.451 | 0.286 | 0.262 | 0.426 | 0.185 | 0.204 | 0.438 | 0.225 | 0.229 |
| 2000 | 1.5 | nan | nan | 0.444 | nan | nan | 0.444 | nan | nan | 0.444 | nan | nan | 0.444 | nan | nan |
| 5000 | 1.8 | nan | nan | 0.361 | nan | nan | 0.377 | nan | nan | 0.426 | nan | nan | 0.4 | nan | nan |
| 10000 | 2.7 | 2.8 | 2.9 | 0.157 | 0.046 | 0.046 | 0.122 | 0.055 | 0.038 | 0.111 | 0.056 | 0.037 | 0.117 | 0.055 | 0.037 |
| 20000 | 4.7 | nan | nan | 0.046 | nan | nan | 0.07 | nan | nan | 0.074 | nan | nan | 0.072 | nan | nan |
| 30000 | 9.1 | 9.3 | nan | 0.009 | 0.009 | nan | 0.018 | 0.0 | nan | 0.019 | 0.0 | nan | 0.018 | 0 | nan |
| 59000 | 17.5 | nan | nan | 0.019 | nan | nan | 0.036 | nan | nan | 0.037 | nan | nan | 0.036 | nan | nan |


---
## Model: deepseek-r1-1.5B-128k, File Extension: html, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 1.4 | 1.4 | 1.4 | 0.481 | 0.407 | 0.38 | 0.467 | 0.308 | 0.276 | 0.259 | 0.148 | 0.148 | 0.333 | 0.2 | 0.193 |
| 10000 | 2.6 | 2.7 | 3.0 | 0.213 | 0.093 | 0.111 | 0.262 | 0.107 | 0.096 | 0.315 | 0.111 | 0.093 | 0.286 | 0.109 | 0.094 |
| 30000 | 10.2 | 9.4 | nan | 0.0 | 0.019 | nan | 0.0 | 0.036 | nan | 0.0 | 0.037 | nan | 0 | 0.036 | nan |


---
## Model: deepseek-r1-1.5B-128k, File Extension: json, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 1.4 | 1.6 | 1.7 | 0.472 | 0.389 | 0.389 | 0.465 | 0.3 | 0.333 | 0.37 | 0.167 | 0.222 | 0.412 | 0.214 | 0.267 |
| 10000 | 2.6 | 2.6 | 2.9 | 0.167 | 0.093 | 0.065 | 0.2 | 0.121 | 0.102 | 0.222 | 0.13 | 0.111 | 0.211 | 0.125 | 0.106 |
| 30000 | 7.9 | nan | nan | 0.009 | nan | nan | 0.0 | nan | nan | 0.0 | nan | nan | 0 | nan | nan |


---
## Model: deepseek-r1-1.5B-128k, File Extension: txt, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 1.4 | 1.8 | 1.6 | 0.519 | 0.407 | 0.435 | 0.516 | 0.407 | 0.431 | 0.593 | 0.407 | 0.407 | 0.552 | 0.407 | 0.419 |
| 2000 | 1.5 | 2.0 | 1.9 | 0.389 | 0.315 | 0.269 | 0.412 | 0.3 | 0.255 | 0.519 | 0.278 | 0.241 | 0.459 | 0.288 | 0.248 |
| 5000 | 1.7 | 2.6 | 2.5 | 0.343 | 0.213 | 0.185 | 0.361 | 0.196 | 0.185 | 0.407 | 0.185 | 0.185 | 0.383 | 0.19 | 0.185 |
| 10000 | 2.9 | 2.5 | 3.1 | 0.185 | 0.074 | 0.056 | 0.196 | 0.129 | 0.056 | 0.204 | 0.148 | 0.056 | 0.2 | 0.138 | 0.056 |
| 20000 | 4.9 | 4.9 | 5.1 | 0.056 | 0.046 | 0.019 | 0.071 | 0.038 | 0.036 | 0.074 | 0.037 | 0.037 | 0.073 | 0.037 | 0.036 |
| 30000 | 8.4 | 8.4 | 9.8 | 0.009 | 0.009 | 0.019 | 0.018 | 0.018 | 0.036 | 0.019 | 0.019 | 0.037 | 0.018 | 0.018 | 0.036 |
| 59000 | 12.0 | 15.2 | 14.0 | 0.0 | 0.019 | 0.056 | 0.0 | 0.036 | 0.1 | 0.0 | 0.037 | 0.111 | 0 | 0.036 | 0.105 |


---
## Model: deepseek-r1-1.5B-128k, File Extension: xml, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 1.5 | 1.6 | 1.5 | 0.528 | 0.37 | 0.407 | 0.525 | 0.354 | 0.375 | 0.574 | 0.315 | 0.278 | 0.549 | 0.333 | 0.319 |
| 2000 | 1.4 | 1.6 | 1.6 | 0.528 | 0.259 | 0.269 | 0.525 | 0.268 | 0.264 | 0.574 | 0.278 | 0.259 | 0.549 | 0.273 | 0.262 |
| 5000 | 1.7 | 2.1 | 2.0 | 0.37 | 0.213 | 0.213 | 0.379 | 0.17 | 0.208 | 0.407 | 0.148 | 0.204 | 0.393 | 0.158 | 0.206 |
| 10000 | 2.5 | 2.4 | 2.5 | 0.176 | 0.083 | 0.037 | 0.193 | 0.105 | 0.054 | 0.204 | 0.111 | 0.056 | 0.198 | 0.108 | 0.055 |
| 20000 | 4.3 | 4.5 | 4.7 | 0.065 | 0.065 | 0.046 | 0.073 | 0.073 | 0.038 | 0.074 | 0.074 | 0.037 | 0.073 | 0.073 | 0.037 |
| 30000 | 6.7 | 6.9 | 7.7 | 0.0 | 0.009 | 0.009 | 0.0 | 0.018 | 0.0 | 0.0 | 0.019 | 0.0 | 0 | 0.018 | 0 |
| 59000 | 13.4 | 16.7 | 17.6 | 0.0 | 0.028 | 0.074 | 0.0 | 0.053 | 0.129 | 0.0 | 0.056 | 0.148 | 0 | 0.054 | 0.138 |


---
## Model: deepseek-r1-1.5B-128k, File Extension: yaml, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 1.6 | 1.6 | 1.6 | 0.528 | 0.417 | 0.343 | 0.526 | 0.408 | 0.293 | 0.556 | 0.37 | 0.222 | 0.541 | 0.388 | 0.253 |
| 10000 | 2.5 | 2.5 | 2.7 | 0.176 | 0.065 | 0.093 | 0.203 | 0.073 | 0.077 | 0.222 | 0.074 | 0.074 | 0.212 | 0.073 | 0.075 |
| 30000 | 8.3 | 7.8 | nan | 0.0 | 0.028 | nan | 0.0 | 0.0 | nan | 0.0 | 0.0 | nan | 0 | 0 | nan |


---
## Model: deepseek-r1-8B-128k, File Extension: csv, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 2.4 | 2.9 | 2.8 | 0.648 | 0.556 | 0.509 | 0.66 | 0.583 | 0.511 | 0.611 | 0.389 | 0.426 | 0.635 | 0.467 | 0.465 |
| 2000 | 2.6 | nan | nan | 0.713 | nan | nan | 0.709 | nan | nan | 0.722 | nan | nan | 0.716 | nan | nan |
| 5000 | 3.6 | nan | nan | 0.694 | nan | nan | 0.684 | nan | nan | 0.722 | nan | nan | 0.703 | nan | nan |
| 10000 | 5.6 | 6.0 | 6.1 | 0.676 | 0.491 | 0.509 | 0.651 | 0.489 | 0.51 | 0.759 | 0.407 | 0.481 | 0.701 | 0.444 | 0.495 |
| 20000 | 11.3 | nan | nan | 0.537 | nan | nan | 0.529 | nan | nan | 0.667 | nan | nan | 0.59 | nan | nan |
| 30000 | 19.9 | 20.7 | nan | 0.519 | 0.333 | nan | 0.515 | 0.339 | nan | 0.648 | 0.352 | nan | 0.574 | 0.345 | nan |


---
## Model: deepseek-r1-8B-128k, File Extension: html, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 2.3 | 2.3 | 2.4 | 0.62 | 0.528 | 0.546 | 0.783 | 0.571 | 0.609 | 0.333 | 0.222 | 0.259 | 0.468 | 0.32 | 0.364 |
| 10000 | 5.6 | 5.7 | 6.0 | 0.685 | 0.62 | 0.491 | 0.656 | 0.618 | 0.49 | 0.778 | 0.63 | 0.463 | 0.712 | 0.624 | 0.476 |
| 30000 | 19.5 | 19.8 | nan | 0.583 | 0.454 | nan | 0.569 | 0.456 | nan | 0.685 | 0.481 | nan | 0.622 | 0.468 | nan |


---
## Model: deepseek-r1-8B-128k, File Extension: json, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 2.4 | 2.8 | 2.9 | 0.685 | 0.537 | 0.509 | 0.692 | 0.556 | 0.512 | 0.667 | 0.37 | 0.407 | 0.679 | 0.444 | 0.454 |
| 10000 | 5.9 | 6.0 | 6.4 | 0.657 | 0.528 | 0.528 | 0.639 | 0.533 | 0.532 | 0.722 | 0.444 | 0.463 | 0.678 | 0.485 | 0.495 |
| 30000 | 20.1 | nan | nan | 0.519 | nan | nan | 0.516 | nan | nan | 0.611 | nan | nan | 0.559 | nan | nan |


---
## Model: deepseek-r1-8B-128k, File Extension: txt, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 2.5 | 2.8 | 3.1 | 0.713 | 0.602 | 0.593 | 0.677 | 0.596 | 0.586 | 0.815 | 0.63 | 0.63 | 0.739 | 0.613 | 0.607 |
| 2000 | 2.9 | 3.5 | 3.4 | 0.713 | 0.593 | 0.565 | 0.667 | 0.583 | 0.556 | 0.852 | 0.648 | 0.648 | 0.748 | 0.614 | 0.598 |
| 5000 | 3.9 | 4.5 | 4.5 | 0.602 | 0.602 | 0.481 | 0.587 | 0.593 | 0.483 | 0.685 | 0.648 | 0.519 | 0.632 | 0.619 | 0.5 |
| 10000 | 6.0 | 7.0 | 6.8 | 0.62 | 0.472 | 0.426 | 0.603 | 0.474 | 0.431 | 0.704 | 0.5 | 0.463 | 0.65 | 0.486 | 0.446 |
| 20000 | 10.9 | 12.5 | 12.5 | 0.546 | 0.38 | 0.398 | 0.536 | 0.39 | 0.415 | 0.685 | 0.426 | 0.5 | 0.602 | 0.407 | 0.454 |
| 30000 | 15.0 | 20.4 | 20.2 | 0.509 | 0.352 | 0.435 | 0.507 | 0.357 | 0.441 | 0.648 | 0.37 | 0.481 | 0.569 | 0.364 | 0.46 |
| 59000 | 14.5 | 32.9 | 29.1 | 0.37 | 0.259 | 0.241 | 0.41 | 0.29 | 0.288 | 0.593 | 0.333 | 0.352 | 0.485 | 0.31 | 0.317 |


---
## Model: deepseek-r1-8B-128k, File Extension: xml, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 2.5 | 2.9 | 2.8 | 0.713 | 0.611 | 0.62 | 0.672 | 0.611 | 0.614 | 0.833 | 0.611 | 0.648 | 0.744 | 0.611 | 0.631 |
| 2000 | 3.0 | 3.0 | 2.9 | 0.676 | 0.611 | 0.574 | 0.651 | 0.603 | 0.569 | 0.759 | 0.648 | 0.611 | 0.701 | 0.625 | 0.589 |
| 5000 | 3.8 | 3.9 | 4.0 | 0.648 | 0.574 | 0.481 | 0.625 | 0.567 | 0.481 | 0.741 | 0.63 | 0.481 | 0.678 | 0.596 | 0.481 |
| 10000 | 5.8 | 6.2 | 6.2 | 0.62 | 0.454 | 0.444 | 0.6 | 0.449 | 0.444 | 0.722 | 0.407 | 0.444 | 0.655 | 0.427 | 0.444 |
| 20000 | 11.0 | 12.0 | 12.0 | 0.546 | 0.343 | 0.407 | 0.537 | 0.333 | 0.4 | 0.667 | 0.315 | 0.37 | 0.595 | 0.324 | 0.385 |
| 30000 | 16.9 | 20.1 | 20.4 | 0.463 | 0.407 | 0.361 | 0.472 | 0.396 | 0.373 | 0.63 | 0.352 | 0.407 | 0.54 | 0.373 | 0.389 |
| 59000 | 23.6 | 35.6 | 34.6 | 0.38 | 0.25 | 0.194 | 0.408 | 0.304 | 0.238 | 0.537 | 0.389 | 0.278 | 0.464 | 0.341 | 0.256 |


---
## Model: deepseek-r1-8B-128k, File Extension: yaml, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 2.5 | 2.9 | 2.7 | 0.694 | 0.602 | 0.593 | 0.684 | 0.612 | 0.6 | 0.722 | 0.556 | 0.556 | 0.703 | 0.583 | 0.577 |
| 10000 | 6.0 | 6.2 | 6.6 | 0.63 | 0.565 | 0.5 | 0.617 | 0.578 | 0.5 | 0.685 | 0.481 | 0.463 | 0.649 | 0.525 | 0.481 |
| 30000 | 19.3 | 20.5 | nan | 0.519 | 0.398 | nan | 0.514 | 0.396 | nan | 0.667 | 0.389 | nan | 0.581 | 0.393 | nan |


---
## Model: llama3.1-8B-128k, File Extension: csv, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.2 | 0.2 | 0.704 | 0.639 | 0.62 | 0.72 | 0.692 | 0.667 | 0.667 | 0.5 | 0.481 | 0.692 | 0.581 | 0.559 |
| 10000 | 2.4 | 2.5 | 2.5 | 0.593 | 0.537 | 0.546 | 0.56 | 0.53 | 0.537 | 0.87 | 0.648 | 0.667 | 0.681 | 0.583 | 0.595 |
| 30000 | 15.0 | 15.2 | nan | 0.426 | 0.315 | nan | 0.455 | 0.368 | nan | 0.741 | 0.519 | nan | 0.563 | 0.431 | nan |


---
## Model: llama3.1-8B-128k, File Extension: html, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.1 | 0.1 | 0.1 | 0.639 | 0.593 | 0.62 | 0.857 | 0.778 | 0.81 | 0.333 | 0.259 | 0.315 | 0.48 | 0.389 | 0.453 |
| 10000 | 2.4 | 2.4 | 2.4 | 0.63 | 0.611 | 0.593 | 0.603 | 0.597 | 0.583 | 0.759 | 0.685 | 0.648 | 0.672 | 0.638 | 0.614 |
| 30000 | 14.8 | nan | nan | 0.537 | nan | nan | 0.524 | nan | nan | 0.796 | nan | nan | 0.632 | nan | nan |


---
## Model: llama3.1-8B-128k, File Extension: json, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.2 | 0.2 | 0.713 | 0.602 | 0.657 | 0.683 | 0.657 | 0.698 | 0.796 | 0.426 | 0.556 | 0.735 | 0.517 | 0.619 |
| 10000 | 2.6 | 2.5 | 2.5 | 0.509 | 0.5 | 0.509 | 0.506 | 0.5 | 0.507 | 0.741 | 0.63 | 0.685 | 0.602 | 0.557 | 0.583 |
| 30000 | 14.6 | nan | nan | 0.37 | nan | nan | 0.417 | nan | nan | 0.648 | nan | nan | 0.507 | nan | nan |


---
## Model: llama3.1-8B-128k, File Extension: txt, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.3 | 0.3 | 0.731 | 0.741 | 0.741 | 0.681 | 0.71 | 0.691 | 0.87 | 0.815 | 0.87 | 0.764 | 0.759 | 0.77 |
| 2000 | 0.3 | 0.4 | 0.4 | 0.63 | 0.648 | 0.63 | 0.588 | 0.618 | 0.592 | 0.87 | 0.778 | 0.833 | 0.701 | 0.689 | 0.692 |
| 5000 | 1.0 | 1.0 | 1.0 | 0.481 | 0.546 | 0.519 | 0.488 | 0.533 | 0.514 | 0.741 | 0.741 | 0.704 | 0.588 | 0.62 | 0.594 |
| 10000 | 2.6 | 2.7 | 2.6 | 0.37 | 0.38 | 0.389 | 0.412 | 0.418 | 0.423 | 0.611 | 0.611 | 0.611 | 0.493 | 0.496 | 0.5 |
| 20000 | 6.4 | 7.4 | 7.6 | 0.407 | 0.296 | 0.333 | 0.434 | 0.359 | 0.364 | 0.611 | 0.519 | 0.444 | 0.508 | 0.424 | 0.4 |
| 30000 | 12.5 | 13.6 | 13.1 | 0.361 | 0.269 | 0.296 | 0.4 | 0.313 | 0.333 | 0.556 | 0.389 | 0.407 | 0.465 | 0.347 | 0.367 |
| 59000 | 16.9 | 29.5 | 24.7 | 0.139 | 0.157 | 0.194 | 0.217 | 0.197 | 0.246 | 0.278 | 0.222 | 0.296 | 0.244 | 0.209 | 0.269 |


---
## Model: llama3.1-8B-128k, File Extension: xml, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.3 | 0.3 | 0.731 | 0.769 | 0.769 | 0.687 | 0.754 | 0.754 | 0.852 | 0.796 | 0.796 | 0.76 | 0.775 | 0.775 |
| 2000 | 0.6 | 0.4 | 0.4 | 0.676 | 0.667 | 0.694 | 0.627 | 0.645 | 0.648 | 0.87 | 0.741 | 0.852 | 0.729 | 0.69 | 0.736 |
| 5000 | 1.0 | 1.1 | 1.1 | 0.528 | 0.565 | 0.602 | 0.518 | 0.548 | 0.575 | 0.796 | 0.741 | 0.778 | 0.628 | 0.63 | 0.661 |
| 10000 | 2.6 | 2.7 | 2.6 | 0.417 | 0.417 | 0.444 | 0.444 | 0.44 | 0.463 | 0.667 | 0.611 | 0.685 | 0.533 | 0.512 | 0.552 |
| 20000 | 6.9 | 7.8 | 8.2 | 0.389 | 0.296 | 0.269 | 0.427 | 0.359 | 0.329 | 0.648 | 0.519 | 0.444 | 0.515 | 0.424 | 0.378 |
| 30000 | 10.9 | 14.2 | 14.5 | 0.38 | 0.231 | 0.315 | 0.413 | 0.29 | 0.368 | 0.574 | 0.37 | 0.519 | 0.481 | 0.325 | 0.431 |
| 59000 | 14.6 | 28.7 | 27.6 | 0.361 | 0.185 | 0.167 | 0.373 | 0.196 | 0.243 | 0.407 | 0.204 | 0.315 | 0.389 | 0.2 | 0.274 |


---
## Model: llama3.1-8B-128k, File Extension: yaml, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.2 | 0.2 | 0.731 | 0.676 | 0.667 | 0.705 | 0.667 | 0.673 | 0.796 | 0.704 | 0.648 | 0.748 | 0.685 | 0.66 |
| 10000 | 2.6 | 2.7 | 2.6 | 0.444 | 0.407 | 0.407 | 0.463 | 0.439 | 0.432 | 0.704 | 0.667 | 0.593 | 0.559 | 0.529 | 0.5 |
| 30000 | 13.6 | 14.7 | nan | 0.324 | 0.231 | nan | 0.383 | 0.307 | nan | 0.574 | 0.426 | nan | 0.459 | 0.357 | nan |


---
## Model: llama3.2-1B-128k, File Extension: csv, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.2 | 0.2 | 0.269 | 0.194 | 0.222 | 0.342 | 0.246 | 0.266 | 0.5 | 0.296 | 0.315 | 0.406 | 0.269 | 0.288 |
| 2000 | 0.2 | nan | nan | 0.333 | nan | nan | 0.393 | nan | nan | 0.611 | nan | nan | 0.478 | nan | nan |
| 5000 | 0.4 | nan | nan | 0.296 | nan | nan | 0.351 | nan | nan | 0.481 | nan | nan | 0.406 | nan | nan |
| 10000 | 0.9 | 0.9 | 0.9 | 0.259 | 0.157 | 0.176 | 0.283 | 0.239 | 0.239 | 0.315 | 0.315 | 0.296 | 0.298 | 0.272 | 0.264 |
| 20000 | 2.9 | nan | nan | 0.241 | nan | nan | 0.182 | nan | nan | 0.148 | nan | nan | 0.163 | nan | nan |
| 30000 | 8.3 | 6.6 | nan | 0.13 | 0.083 | nan | 0.045 | 0.131 | nan | 0.037 | 0.148 | nan | 0.041 | 0.139 | nan |
| 59000 | 25.9 | nan | nan | 0.009 | nan | nan | 0.018 | nan | nan | 0.019 | nan | nan | 0.018 | nan | nan |


---
## Model: llama3.2-1B-128k, File Extension: html, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.1 | 0.1 | 0.1 | 0.157 | 0.139 | 0.148 | 0.224 | 0.145 | 0.172 | 0.278 | 0.148 | 0.185 | 0.248 | 0.147 | 0.179 |
| 10000 | 0.9 | 0.9 | 0.9 | 0.269 | 0.222 | 0.176 | 0.338 | 0.292 | 0.246 | 0.481 | 0.389 | 0.315 | 0.397 | 0.333 | 0.276 |
| 30000 | 6.7 | 6.4 | nan | 0.231 | 0.083 | nan | 0.146 | 0.143 | nan | 0.111 | 0.167 | nan | 0.126 | 0.154 | nan |


---
## Model: llama3.2-1B-128k, File Extension: json, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.2 | 0.2 | 0.343 | 0.204 | 0.231 | 0.404 | 0.265 | 0.301 | 0.667 | 0.333 | 0.407 | 0.503 | 0.295 | 0.346 |
| 10000 | 0.9 | 0.9 | 0.9 | 0.259 | 0.13 | 0.213 | 0.217 | 0.206 | 0.208 | 0.185 | 0.259 | 0.204 | 0.2 | 0.23 | 0.206 |
| 30000 | 7.5 | nan | nan | 0.102 | nan | nan | 0.022 | nan | nan | 0.019 | nan | nan | 0.02 | nan | nan |


---
## Model: llama3.2-1B-128k, File Extension: txt, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.1 | 0.2 | 0.2 | 0.352 | 0.324 | 0.278 | 0.405 | 0.383 | 0.346 | 0.63 | 0.574 | 0.5 | 0.493 | 0.459 | 0.409 |
| 2000 | 0.2 | 0.2 | 0.2 | 0.278 | 0.296 | 0.287 | 0.342 | 0.347 | 0.358 | 0.481 | 0.463 | 0.537 | 0.4 | 0.397 | 0.43 |
| 5000 | 0.3 | 0.4 | 0.4 | 0.343 | 0.167 | 0.157 | 0.16 | 0.243 | 0.239 | 0.074 | 0.315 | 0.315 | 0.101 | 0.274 | 0.272 |
| 10000 | 1.4 | 1.0 | 0.9 | 0.093 | 0.167 | 0.213 | 0.042 | 0.219 | 0.196 | 0.037 | 0.259 | 0.185 | 0.039 | 0.237 | 0.19 |
| 20000 | 3.2 | 5.3 | 5.6 | 0.074 | 0.028 | 0.093 | 0.0 | 0.036 | 0.042 | 0.0 | 0.037 | 0.037 | 0 | 0.037 | 0.039 |
| 30000 | 4.9 | 8.7 | 10.4 | 0.009 | 0.0 | 0.009 | 0.018 | 0.0 | 0.018 | 0.019 | 0.0 | 0.019 | 0.018 | 0 | 0.018 |
| 59000 | 4.5 | 18.1 | 17.9 | 0.009 | 0.0 | 0.028 | 0.018 | 0.0 | 0.053 | 0.019 | 0.0 | 0.056 | 0.018 | 0 | 0.054 |


---
## Model: llama3.2-1B-128k, File Extension: xml, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.2 | 0.2 | 0.324 | 0.333 | 0.287 | 0.383 | 0.39 | 0.351 | 0.574 | 0.593 | 0.5 | 0.459 | 0.471 | 0.412 |
| 2000 | 0.1 | 0.2 | 0.2 | 0.315 | 0.315 | 0.25 | 0.372 | 0.365 | 0.32 | 0.537 | 0.5 | 0.444 | 0.439 | 0.422 | 0.372 |
| 5000 | 0.3 | 0.4 | 0.4 | 0.25 | 0.204 | 0.204 | 0.286 | 0.289 | 0.271 | 0.333 | 0.407 | 0.352 | 0.308 | 0.338 | 0.306 |
| 10000 | 1.3 | 1.0 | 0.9 | 0.222 | 0.139 | 0.167 | 0.159 | 0.217 | 0.19 | 0.13 | 0.278 | 0.204 | 0.143 | 0.244 | 0.196 |
| 20000 | 6.1 | 4.5 | 4.3 | 0.0 | 0.019 | 0.111 | 0.0 | 0.036 | 0.023 | 0.0 | 0.037 | 0.019 | 0 | 0.036 | 0.02 |
| 30000 | 7.9 | 10.6 | 10.7 | 0.019 | 0.0 | 0.019 | 0.036 | 0.0 | 0.036 | 0.037 | 0.0 | 0.037 | 0.036 | 0 | 0.036 |
| 59000 | 11.0 | 20.0 | 21.0 | 0.019 | 0.009 | 0.019 | 0.036 | 0.018 | 0.036 | 0.037 | 0.019 | 0.037 | 0.036 | 0.018 | 0.036 |


---
## Model: llama3.2-1B-128k, File Extension: yaml, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.1 | 0.2 | 0.2 | 0.37 | 0.278 | 0.306 | 0.42 | 0.333 | 0.367 | 0.685 | 0.444 | 0.537 | 0.521 | 0.381 | 0.436 |
| 10000 | 1.1 | 0.9 | 0.9 | 0.296 | 0.139 | 0.148 | 0.107 | 0.217 | 0.183 | 0.056 | 0.278 | 0.204 | 0.073 | 0.244 | 0.193 |
| 30000 | 10.4 | 9.8 | nan | 0.019 | 0.0 | nan | 0.036 | 0.0 | nan | 0.037 | 0.0 | nan | 0.036 | 0 | nan |


---
## Model: llama3.2-3B-128k, File Extension: csv, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.1 | 0.1 | 0.1 | 0.685 | 0.583 | 0.583 | 0.738 | 0.765 | 0.714 | 0.574 | 0.241 | 0.278 | 0.646 | 0.366 | 0.4 |
| 10000 | 1.5 | 1.6 | 1.7 | 0.435 | 0.361 | 0.38 | 0.441 | 0.368 | 0.377 | 0.481 | 0.389 | 0.37 | 0.46 | 0.378 | 0.374 |
| 30000 | 10.7 | 10.7 | nan | 0.296 | 0.194 | nan | 0.077 | 0.077 | nan | 0.037 | 0.056 | nan | 0.05 | 0.065 | nan |


---
## Model: llama3.2-3B-128k, File Extension: html, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.1 | 0.1 | 0.1 | 0.63 | 0.546 | 0.565 | 1.0 | 1.0 | 0.889 | 0.259 | 0.093 | 0.148 | 0.412 | 0.169 | 0.254 |
| 10000 | 1.5 | 1.7 | 1.6 | 0.565 | 0.444 | 0.435 | 0.556 | 0.452 | 0.434 | 0.648 | 0.519 | 0.426 | 0.598 | 0.483 | 0.43 |
| 30000 | 9.8 | nan | nan | 0.296 | nan | nan | 0.176 | nan | nan | 0.111 | nan | nan | 0.136 | nan | nan |


---
## Model: llama3.2-3B-128k, File Extension: json, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.1 | 0.1 | 0.1 | 0.667 | 0.546 | 0.62 | 0.714 | 0.647 | 0.842 | 0.556 | 0.204 | 0.296 | 0.625 | 0.31 | 0.438 |
| 10000 | 1.6 | 1.7 | 1.7 | 0.407 | 0.324 | 0.407 | 0.396 | 0.289 | 0.391 | 0.352 | 0.241 | 0.333 | 0.373 | 0.263 | 0.36 |
| 30000 | 9.5 | nan | nan | 0.278 | nan | nan | 0.038 | nan | nan | 0.019 | nan | nan | 0.025 | nan | nan |


---
## Model: llama3.2-3B-128k, File Extension: txt, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.2 | 0.2 | 0.611 | 0.731 | 0.713 | 0.594 | 0.805 | 0.767 | 0.704 | 0.611 | 0.611 | 0.644 | 0.695 | 0.68 |
| 2000 | 0.3 | 0.3 | 0.3 | 0.435 | 0.583 | 0.528 | 0.455 | 0.567 | 0.522 | 0.648 | 0.704 | 0.667 | 0.534 | 0.628 | 0.585 |
| 5000 | 0.7 | 0.8 | 0.7 | 0.407 | 0.287 | 0.333 | 0.407 | 0.298 | 0.333 | 0.407 | 0.315 | 0.333 | 0.407 | 0.306 | 0.333 |
| 10000 | 1.6 | 1.7 | 3.8 | 0.259 | 0.046 | 0.093 | 0.24 | 0.07 | 0.077 | 0.222 | 0.074 | 0.074 | 0.231 | 0.072 | 0.075 |
| 20000 | 7.2 | 10.7 | 12.6 | 0.139 | 0.102 | 0.0 | 0.047 | 0.169 | 0.0 | 0.037 | 0.204 | 0.0 | 0.041 | 0.185 | 0 |
| 30000 | 14.9 | 17.8 | 18.5 | 0.009 | 0.037 | 0.009 | 0.0 | 0.069 | 0.018 | 0.0 | 0.074 | 0.019 | 0 | 0.071 | 0.018 |
| 59000 | 20.3 | 30.9 | 30.2 | 0.0 | 0.019 | 0.037 | 0.0 | 0.036 | 0.069 | 0.0 | 0.037 | 0.074 | 0 | 0.036 | 0.071 |


---
## Model: llama3.2-3B-128k, File Extension: xml, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.2 | 0.2 | 0.676 | 0.759 | 0.731 | 0.661 | 0.833 | 0.755 | 0.722 | 0.648 | 0.685 | 0.69 | 0.729 | 0.718 |
| 2000 | 0.2 | 0.3 | 0.3 | 0.491 | 0.611 | 0.611 | 0.493 | 0.597 | 0.594 | 0.685 | 0.685 | 0.704 | 0.574 | 0.638 | 0.644 |
| 5000 | 0.6 | 0.7 | 0.8 | 0.38 | 0.361 | 0.343 | 0.386 | 0.368 | 0.345 | 0.407 | 0.389 | 0.352 | 0.396 | 0.378 | 0.349 |
| 10000 | 1.7 | 1.9 | 3.0 | 0.315 | 0.083 | 0.056 | 0.321 | 0.075 | 0.056 | 0.333 | 0.074 | 0.056 | 0.327 | 0.075 | 0.056 |
| 20000 | 6.6 | 12.2 | 12.4 | 0.111 | 0.019 | 0.028 | 0.125 | 0.036 | 0.053 | 0.13 | 0.037 | 0.056 | 0.127 | 0.036 | 0.054 |
| 30000 | 15.5 | 16.6 | 13.3 | 0.046 | 0.037 | 0.037 | 0.07 | 0.069 | 0.069 | 0.074 | 0.074 | 0.074 | 0.072 | 0.071 | 0.071 |
| 59000 | 20.4 | 21.6 | 21.0 | 0.0 | 0.046 | 0.185 | 0.0 | 0.085 | 0.27 | 0.0 | 0.093 | 0.37 | 0 | 0.088 | 0.312 |


---
## Model: llama3.2-3B-128k, File Extension: yaml, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.1 | 0.1 | 0.1 | 0.741 | 0.62 | 0.667 | 0.771 | 0.842 | 0.875 | 0.685 | 0.296 | 0.389 | 0.725 | 0.438 | 0.538 |
| 10000 | 1.5 | 1.5 | 1.7 | 0.315 | 0.324 | 0.389 | 0.315 | 0.314 | 0.312 | 0.315 | 0.296 | 0.185 | 0.315 | 0.305 | 0.233 |
| 30000 | 11.5 | 13.3 | nan | 0.083 | 0.056 | nan | 0.0 | 0.02 | nan | 0.0 | 0.019 | nan | 0 | 0.019 | nan |


---
## Model: phi3-14B-q4-medium-128k, File Extension: csv, Context Type: token

| Noise Level | Mean Duration(s) | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- |
|  | English | English | English | English | English |
| --- | --- | --- | --- | --- | --- |
| 1000 | 4.1 | 0.63 | 0.64 | 0.593 | 0.615 |
| 10000 | 12.7 | 0.667 | 0.667 | 0.667 | 0.667 |
| 30000 | 53.7 | 0.667 | 0.86 | 0.685 | 0.763 |


---
## Model: phi3-14B-q4-medium-128k, File Extension: txt, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 4.1 | 4.1 | 3.9 | 0.741 | 0.694 | 0.704 | 0.75 | 0.667 | 0.72 | 0.722 | 0.778 | 0.667 | 0.736 | 0.718 | 0.692 |
| 2000 | 4.9 | 4.7 | 4.6 | 0.75 | 0.704 | 0.676 | 0.765 | 0.696 | 0.711 | 0.722 | 0.722 | 0.593 | 0.743 | 0.709 | 0.646 |
| 5000 | 7.9 | 7.3 | 6.9 | 0.713 | 0.694 | 0.62 | 0.745 | 0.756 | 0.667 | 0.648 | 0.574 | 0.481 | 0.693 | 0.653 | 0.559 |
| 10000 | 13.5 | 12.9 | 11.8 | 0.704 | 0.657 | 0.657 | 0.72 | 0.689 | 0.698 | 0.667 | 0.574 | 0.556 | 0.692 | 0.626 | 0.619 |
| 20000 | 30.6 | 28.7 | 26.3 | 0.639 | 0.639 | 0.574 | 0.619 | 0.66 | 0.587 | 0.722 | 0.574 | 0.5 | 0.667 | 0.614 | 0.54 |
| 30000 | 57.0 | 51.2 | 48.4 | 0.546 | 0.63 | 0.565 | 0.54 | 0.635 | 0.574 | 0.63 | 0.611 | 0.5 | 0.581 | 0.623 | 0.535 |
| 59000 | 175.9 | 155.6 | 146.6 | 0.472 | 0.519 | 0.528 | 0.478 | 0.52 | 0.532 | 0.611 | 0.481 | 0.463 | 0.537 | 0.5 | 0.495 |


---
## Model: phi3-14B-q4-medium-128k, File Extension: xml, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 4.0 | 4.3 | 3.8 | 0.741 | 0.685 | 0.694 | 0.717 | 0.667 | 0.691 | 0.796 | 0.741 | 0.704 | 0.754 | 0.702 | 0.697 |
| 2000 | 4.8 | 4.7 | 4.6 | 0.731 | 0.722 | 0.731 | 0.745 | 0.74 | 0.755 | 0.704 | 0.685 | 0.685 | 0.724 | 0.712 | 0.718 |
| 5000 | 7.7 | 7.2 | 7.1 | 0.722 | 0.759 | 0.676 | 0.731 | 0.792 | 0.732 | 0.704 | 0.704 | 0.556 | 0.717 | 0.745 | 0.632 |
| 10000 | 12.7 | 12.2 | 11.7 | 0.685 | 0.657 | 0.611 | 0.7 | 0.681 | 0.667 | 0.648 | 0.593 | 0.444 | 0.673 | 0.634 | 0.533 |
| 20000 | 28.6 | 26.8 | 25.5 | 0.657 | 0.574 | 0.546 | 0.655 | 0.591 | 0.556 | 0.667 | 0.481 | 0.463 | 0.661 | 0.531 | 0.505 |
| 30000 | 52.2 | 48.3 | 45.9 | 0.63 | 0.602 | 0.546 | 0.617 | 0.608 | 0.561 | 0.685 | 0.574 | 0.426 | 0.649 | 0.59 | 0.484 |
| 59000 | 161.6 | 146.1 | 139.9 | 0.528 | 0.528 | 0.519 | 0.522 | 0.529 | 0.521 | 0.667 | 0.5 | 0.463 | 0.585 | 0.514 | 0.49 |


---
## Model: qwen2.5-7B-128k, File Extension: csv, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.2 | 0.2 | 0.759 | 0.667 | 0.694 | 0.85 | 0.8 | 0.862 | 0.63 | 0.444 | 0.463 | 0.723 | 0.571 | 0.602 |
| 10000 | 2.3 | 2.4 | 2.3 | 0.824 | 0.741 | 0.713 | 0.843 | 0.842 | 0.78 | 0.796 | 0.593 | 0.593 | 0.819 | 0.696 | 0.674 |
| 30000 | 13.7 | 13.8 | nan | 0.704 | 0.694 | nan | 0.683 | 0.756 | nan | 0.759 | 0.574 | nan | 0.719 | 0.653 | nan |


---
## Model: qwen2.5-7B-128k, File Extension: html, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.1 | 0.1 | 0.1 | 0.676 | 0.611 | 0.639 | 1.0 | 0.875 | 0.941 | 0.352 | 0.259 | 0.296 | 0.521 | 0.4 | 0.451 |
| 10000 | 2.1 | 2.1 | 2.1 | 0.843 | 0.787 | 0.741 | 0.849 | 0.83 | 0.81 | 0.833 | 0.722 | 0.63 | 0.841 | 0.772 | 0.708 |
| 30000 | 12.4 | nan | nan | 0.778 | nan | nan | 0.768 | nan | nan | 0.796 | nan | nan | 0.782 | nan | nan |


---
## Model: qwen2.5-7B-128k, File Extension: json, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.2 | 0.2 | 0.852 | 0.676 | 0.639 | 0.913 | 0.852 | 0.778 | 0.778 | 0.426 | 0.389 | 0.84 | 0.568 | 0.519 |
| 10000 | 2.2 | 2.3 | 2.3 | 0.824 | 0.731 | 0.759 | 0.857 | 0.805 | 0.818 | 0.778 | 0.611 | 0.667 | 0.816 | 0.695 | 0.735 |
| 30000 | 12.5 | nan | nan | 0.75 | nan | nan | 0.745 | nan | nan | 0.759 | nan | nan | 0.752 | nan | nan |


---
## Model: qwen2.5-7B-128k, File Extension: txt, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.2 | 0.3 | 0.833 | 0.824 | 0.833 | 0.846 | 0.872 | 0.86 | 0.815 | 0.759 | 0.796 | 0.83 | 0.812 | 0.827 |
| 2000 | 0.4 | 0.4 | 0.4 | 0.833 | 0.806 | 0.833 | 0.846 | 0.837 | 0.875 | 0.815 | 0.759 | 0.778 | 0.83 | 0.796 | 0.824 |
| 5000 | 1.0 | 0.9 | 1.0 | 0.796 | 0.722 | 0.75 | 0.82 | 0.816 | 0.814 | 0.759 | 0.574 | 0.648 | 0.788 | 0.674 | 0.722 |
| 10000 | 2.3 | 2.2 | 2.3 | 0.769 | 0.694 | 0.657 | 0.796 | 0.784 | 0.743 | 0.722 | 0.537 | 0.481 | 0.757 | 0.637 | 0.584 |
| 20000 | 5.9 | 6.3 | 6.5 | 0.741 | 0.63 | 0.63 | 0.71 | 0.675 | 0.694 | 0.815 | 0.5 | 0.463 | 0.759 | 0.574 | 0.556 |
| 30000 | 8.7 | 11.4 | 11.3 | 0.648 | 0.667 | 0.611 | 0.633 | 0.765 | 0.714 | 0.704 | 0.481 | 0.37 | 0.667 | 0.591 | 0.488 |
| 59000 | 5.8 | 19.2 | 18.8 | 0.38 | 0.63 | 0.546 | 0.386 | 0.684 | 0.6 | 0.407 | 0.481 | 0.278 | 0.396 | 0.565 | 0.38 |


---
## Model: qwen2.5-7B-128k, File Extension: xml, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.4 | 0.3 | 0.861 | 0.861 | 0.787 | 0.855 | 0.915 | 0.83 | 0.87 | 0.796 | 0.722 | 0.862 | 0.851 | 0.772 |
| 2000 | 0.4 | 0.3 | 0.4 | 0.815 | 0.833 | 0.815 | 0.854 | 0.875 | 0.854 | 0.759 | 0.778 | 0.759 | 0.804 | 0.824 | 0.804 |
| 5000 | 0.9 | 0.9 | 0.9 | 0.796 | 0.778 | 0.787 | 0.82 | 0.875 | 0.844 | 0.759 | 0.648 | 0.704 | 0.788 | 0.745 | 0.768 |
| 10000 | 2.3 | 2.2 | 2.2 | 0.815 | 0.667 | 0.639 | 0.827 | 0.75 | 0.742 | 0.796 | 0.5 | 0.426 | 0.811 | 0.6 | 0.541 |
| 20000 | 6.1 | 6.2 | 6.3 | 0.741 | 0.657 | 0.685 | 0.732 | 0.73 | 0.812 | 0.759 | 0.5 | 0.481 | 0.745 | 0.593 | 0.605 |
| 30000 | 9.7 | 11.7 | 11.8 | 0.685 | 0.639 | 0.62 | 0.667 | 0.778 | 0.741 | 0.741 | 0.389 | 0.37 | 0.702 | 0.519 | 0.494 |
| 59000 | 13.3 | 23.6 | 23.0 | 0.472 | 0.602 | 0.565 | 0.474 | 0.69 | 0.667 | 0.5 | 0.37 | 0.259 | 0.486 | 0.482 | 0.373 |


---
## Model: qwen2.5-7B-128k, File Extension: yaml, Context Type: token

| Noise Level | Mean Duration(s) | Mean Duration(s) | Mean Duration(s) | Accuracy | Accuracy | Accuracy | Precision | Precision | Precision | Recall | Recall | Recall | F1 Score | F1 Score | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | English | French | German | English | French | German | English | French | German | English | French | German | English | French | German |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.2 | 0.2 | 0.2 | 0.843 | 0.769 | 0.731 | 0.878 | 0.872 | 0.821 | 0.796 | 0.63 | 0.593 | 0.835 | 0.731 | 0.688 |
| 10000 | 2.2 | 2.2 | 2.2 | 0.796 | 0.722 | 0.685 | 0.833 | 0.786 | 0.778 | 0.741 | 0.611 | 0.519 | 0.784 | 0.688 | 0.622 |
| 30000 | 11.9 | 12.0 | nan | 0.741 | 0.63 | nan | 0.771 | 0.75 | nan | 0.685 | 0.389 | nan | 0.725 | 0.512 | nan |


---
