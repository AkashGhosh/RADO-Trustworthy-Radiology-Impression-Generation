# RADO : Trustworthy Radiology Impression Generation using Safety and Faithfulness-based Preference Optimization
**Accepted in ACM Transactions on Computing for  Healthcare** 

<p align="center">
  <img src="./Image/RADO_final_page-0001.jpg" />
</p>


## Dataset

- **RIB**: Provided in `data.zip`. The dataset contains open-ended medical reasoning questions with a single verifiable answer across **13 languages**.  
  Unzip `data.zip` before running training or evaluation.

- **Hugging Face**: RIB is also available on Hugging Face:  


## Installation

## Repository Structure


## About the Paper
RADO is a novel framework for radiology impression generation that integrates safety, faithfulness, and linguistic refinement rewards for preference optimization. To support robust evaluation, we introduce RIB, a radiologist-curated benchmark of 2,800 annotated CT and MRI findings and impressions across 27 study types. RADO achieves state-of-the-art performance across automatic and human evaluation metrics, demonstrating improved factual consistency, reduced omissions, and higher clinical relevance — advancing the safety and reliability of generative AI in high-stakes medical applications.

## Contributions

- RIB Dataset: 2,800 expert-annotated radiology reports across 27 study types, built with structured clinical guidelines and multi-stage quality control.
- RADO Framework: Expert-informed reward models targeting key failure modes — fabrication, severity misclassification, terminology errors, and omissions — through calibrated safety, faithfulness, and linguistic components.
- Empirical Evaluation: Rigorous assessment combining automated metrics and human evaluation by radiology interns supervised by experienced radiologists, demonstrating improvements in factual consistency, reduced omissions, and clinical relevance.

## Evaluation Results

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Table 3 – RADO Performance Comparison</title>
<style>
  body {
    font-family: 'Times New Roman', Times, serif;
    background: #fff;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 30px 20px;
    color: #000;
  }

  .table-wrapper {
    max-width: 860px;
    width: 100%;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
  }

  thead tr {
    border-top: 2px solid #000;
    border-bottom: 1.5px solid #000;
  }

  thead th {
    padding: 6px 10px;
    text-align: left;
    font-weight: bold;
    font-size: 13.5px;
    white-space: nowrap;
  }

  thead th:not(:first-child) {
    text-align: center;
  }

  tbody tr td {
    padding: 4px 10px;
    text-align: center;
    font-size: 13.5px;
  }

  tbody tr td:first-child {
    text-align: left;
  }

  /* Last row border */
  tbody tr:last-child {
    border-bottom: 2px solid #000;
  }

  /* Section divider after baseline models */
  tr.section-divider td {
    border-top: 1px solid #ccc;
  }

  /* RADO highlighted rows - bold */
  tr.rado-row td {
    font-weight: bold;
  }

  /* Green rows - RADO reward ablation */
  tr.green-row {
    background-color: #d4edda;
  }

  /* Pink rows - RL method ablation */
  tr.pink-row {
    background-color: #fce4ec;
  }

  .caption {
    margin-top: 10px;
    font-size: 12.5px;
    text-align: left;
    max-width: 860px;
    width: 100%;
    line-height: 1.5;
    color: #000;
  }

  .caption strong {
    font-weight: bold;
  }
</style>
</head>
<body>
<div class="table-wrapper">
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Rouge-1</th>
        <th>Rouge-2</th>
        <th>Rouge-L</th>
        <th>Bert Score</th>
        <th>SummaCC</th>
        <th>FactCC</th>
      </tr>
    </thead>
    <tbody>
      <!-- Baselines -->
      <tr>
        <td>GPT-4o(mini)</td>
        <td>38.04</td><td>24.77</td><td>31.54</td><td>0.76</td><td>28.64</td><td>23.26</td>
      </tr>
      <tr>
        <td>GPT-4o</td>
        <td>42.89</td><td>30.19</td><td>37.21</td><td>0.79</td><td>32.21</td><td>47.87</td>
      </tr>
      <tr>
        <td>LLAMA-3-70B</td>
        <td>41.71</td><td>30.49</td><td>37.10</td><td>0.79</td><td>31.89</td><td>47.17</td>
      </tr>
      <tr>
        <td>Qwen-72B</td>
        <td>45.78</td><td>34.70</td><td>41.13</td><td>0.79</td><td>38.01</td><td>53.20</td>
      </tr>
      <tr>
        <td>Phi3-medium-instruct</td>
        <td>26.9</td><td>14.10</td><td>22.28</td><td>0.72</td><td>24.58</td><td>44.20</td>
      </tr>
      <tr>
        <td>Phi3-mini-instruct (10 shot)</td>
        <td>28.92</td><td>15.17</td><td>24.1</td><td>0.73</td><td>25.83</td><td>50.96</td>
      </tr>
      <tr>
        <td>Phi3-mini-instruct(SFT)</td>
        <td>46.18</td><td>39.16</td><td>44.51</td><td>0.80</td><td>51.65</td><td>47.43</td>
      </tr>
      <tr class="rado-row">
        <td><b>RADO(Phi3-mini)</b></td>
        <td>52.08</td><td>45.08</td><td>54.18</td><td>0.83</td><td>51.85</td><td>54.5</td>
      </tr>

      <!-- Qwen section -->
      <tr class="section-divider">
        <td>Qwen-2 1.5B(10 shot)</td>
        <td>26.97</td><td>17.81</td><td>23.06</td><td>0.70</td><td>29.34</td><td>42.92</td>
      </tr>
      <tr>
        <td>Qwen-2 1.5B(SFT)</td>
        <td>52.21</td><td>44.82</td><td>50.50</td><td>0.85</td><td>50.50</td><td>48.61</td>
      </tr>
      <tr class="rado-row">
        <td><b>RADO(Qwen2-1.5B)</b></td>
        <td>55.9</td><td>48.21</td><td>54.18</td><td>0.85</td><td>52.10</td><td>57.41</td>
      </tr>

      <!-- LLAMA 3.2 section -->
      <tr class="section-divider">
        <td>LLAMA 3.2 3B (10 shot)</td>
        <td>29.02</td><td>18.10</td><td>24.79</td><td>0.71</td><td>28.10</td><td>40.05</td>
      </tr>
      <tr>
        <td>LLAMA 3.2 3B (SFT)</td>
        <td>50.76</td><td>43.12</td><td>48.94</td><td>0.84</td><td>52.34</td><td>50.34</td>
      </tr>
      <tr class="rado-row">
        <td><b>RADO(LLAMA 3.2 3B)</b></td>
        <td>56.30</td><td>48.34</td><td>54.24</td><td>0.85</td><td>52.49</td><td>61.43</td>
      </tr>

      <!-- Green rows - reward ablation -->
      <tr class="section-divider green-row">
        <td>RADO(LBR)</td>
        <td>53.2</td><td>45.1</td><td>51.2</td><td>0.84</td><td>51.74</td><td>55.47</td>
      </tr>
      <tr class="green-row">
        <td>RADO(SBR)</td>
        <td>55.1</td><td>46.7</td><td>53.4</td><td>0.85</td><td>51.85</td><td>56.28</td>
      </tr>
      <tr class="green-row">
        <td>RADO(FBR)</td>
        <td>55.31</td><td>47.5</td><td>54.14</td><td>0.85</td><td>51.89</td><td>56.32</td>
      </tr>

      <!-- Pink rows - RL ablation -->
      <tr class="pink-row">
        <td>RADO(RLOO)</td>
        <td>54.16</td><td>46.61</td><td>52.12</td><td>0.84</td><td>51.25</td><td>59.57</td>
      </tr>
      <tr class="pink-row">
        <td>RADO(PPO)</td>
        <td>55.34</td><td>47.8</td><td>54.10</td><td>0.84</td><td>51.56</td><td>60.45</td>
      </tr>
    </tbody>
  </table>
</div>

<p class="caption">
  <strong>Table 3.</strong> Performance comparison of various models with respect to <strong><em>RADO</em></strong>. The impact of different reward models (safety, faithfulness) and reinforcement learning methods (RLOO, PPO) is highlighted in green and pink. Here LBR means linguistic-based rewards, FBR means faithfulness-based rewards and SBR means safety-based Rewards.
</p>

</body>
</html>

