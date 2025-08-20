# A Multimodal Approach to Reconstructing Geological Uncertainties Using a Deep Residual Variational Autoencoder3
Our paper is under review


This repo explains dependencies, training, and testing for every module of our network:

## Dependencies
The implementation has been tested under a Mac OS environment with:
* Python (V3.7)
* Pandas (V2.0.3)
* Sklearn (V1.3.0)
* Scipy (V1.9.1)
* Matplotlib (V3.7.2)
* Seaborn (V0.12.2)
* Gpustat (V1.1)
* keras_application (V.0.8)
* OpenCV (V4.5.1)
* Tensorflow (V2.12.0)
* Numpy (V1.23.0)

Other library versions and operating systems may require minor adjustments.


## Train and Test

To facilitate training and testing, we provide a comprehensive python file: `main.py`. 

The main function, `pretrained`, requires six parameters:

1. **uncertainties_path**: Path where normalized uncertainties are stored.

2. **production_path**: Path where cumulative curves are saved.

3. **history_path**: Path where historical data is stored.

4. **days_txt**: Path to the file containing the selected five days to represent each cumulative curve.

5. **gpu_id**: To utilize multiple GPUs, specify `--gpu_id 0,1,2,3`.

6. **is_some_days**: Option to choose between 2206 days or 5 days for the cumulative curve, you can set `--is_some_days 'someDays', 'allDays'`.


```bash
    pretrained(
        uncertainties_path=uncertainties_file,
        production_path=simulation_file,
        history_path=history_file,
        days_txt=selected_days_file,
        gpu_id=gpu_id,
        is_some_days=days_option,
        path_model_to_save=results_save_path
    )
```


## Acknowledgment
This work was conducted with the support of Energi Simulation and in association with the ongoing Project registered under ANP number 21870-1 as "Desenvolvimento de uma Abordagem para Construção de Modelos Multifidelidades para Reduzir Incertezas e Melhorar Previsão de Produção" (UNICAMP/Shell Brazil/ANP) funded by Shell Brazil, under the ANP R&D levy as "Compromisso de Investimentos com Pesquisa e Desenvolvimento". The authors thank also UNISIM, DE-FEM-UNICAMP, Recod.ai, IC-UNICAMP, and CEPETRO for supporting this work. We also thank CMG, Emerson, and Schlumberger for software licenses.

