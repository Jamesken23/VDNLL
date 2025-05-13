# VDNLL (Vulnerability Detection with Noisy Label Learning)
This repo is a paper on Python implementation: **A Robust Vulnerability Detection Framework by Learning From Noisy Labels**. In this paper, we propose a robust smart contract Vulnerability Detection framework with Noisy Label Learning (VDNLL). It develops a noisy label learning method based on co-training to identify mislabeled samples more accurately, facilitating the learning of robust feature representations. 

# Datasets
In this study, we adopt the widely-studied smart contract dataset [Li et al., 2022](https://ieeexplore.ieee.org/abstract/document/10261219) as the benchmark, which consists of 38,600 contract source codes written with Solidity language. The contract source codes within the benchmark dataset are collected from the real-world Ethereum platform by Li et al. This team collected and organized a total of five open source smart contract datasets, i.e., from \texttt{Dataset\_1} to \texttt{Dataset\_5}. The {Dataset\_2} and {Dataset\_4} only contain smart contracts in bytecode form, which is unsuitable for experimental comparison in RQ2. The reason is that deep learning-based vulnerability detection baseline methods mostly take the contract source code as input. The {Dataset\_3} and {Dataset\_5} are made up of contract source codes, yet these two datasets have small sample sizes. Hence, we chose the {Dataset\_1} as the benchmark dataset, with the number and size of the contract samples being 38,600 and 433.3MB, respectively.
The above benchmark dataset contains thirty types of smart contract vulnerabilities. We focus on evaluating the detection performance of VDNLL in four kinds of vulnerabilities, including reentrancy (RE), transaction order dependency (TOD), locked ether (LE), and suicide contract (SU). These four vulnerabilities hold representative vulnerability characteristics of Ethereum smart contracts. According to the statistics, there are 425 of 2,548, 7,532 of 38,501, 4,240 of 25,404, and 191 of 1,147 vulnerable contracts within the RE, TOD, LE, and SU datasets.

Further instructions on the datasets can be found on [Vulhunter-Dataset1](https://github.com/ContractAudit/VulHunter/tree/main/Dataset1) or [Vulhunter-alternate-Dataset1](https://drive.google.com/drive/folders/1u_nPauSv62jR5EUi3p1DGc_uc5B2FTUP), which is constantly being updated to provide more details.

# Required Packages
> - Python 3.8
> - torch 1.7.1
> - faiss 1.7.1
> - numpy 1.24.3
> - scikit-learn 1.3.2

# Tools
VDNLL is more robust against noisy labels, and can effectively maintain the performance of vulnerability detection in all noisy label settings. To support this, the t-distributed stochastic neighbor embedding (TNSE) [Maaten et al., 2008](https://jmlr.org/papers/v9/vandermaaten08a.html) is applied to visualize the feature representations of the RE dataset. 


In this work, we present the feature representations learned by the GraBit, ConvMHSA, and VDNLL, in the clean and noisy label settings. Compared to existing popular vulnerability detection methods, the feature representations learned by the VDNLL are always more class-discriminative. This proves that our approach can indeed learn robust code feature representations from noisy labels.

We will continue to add and update more details on this work.
