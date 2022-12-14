# Automated Trading Simulation with ARMA Model

This project utilizes the Auto regression moving average model and the twelve data API to simulate an automated trading system in nigh real-time.

for more info on the auto regression moving avarage model (ARMA) visit this [notebook](https://www.kaggle.com/code/henrychibueze/working-principle-of-the-arma-model#ARMA-model) on kaggle I composed.

documentation of twelve data API can be accessed here [here](https://rapidapi.com/twelvedata/api/twelve-data1/).

### DISCLAIMER
This is for educational purpose, use this for actual trading of stocks at your own risk.

### How to Run
1. First clone the repository and run the following command: `pip install -r requirements.txt`.

2. Next create a `.env` file and copy the contents of `.env.example` file to it.

3. Next go to [rapidapi](https://rapidapi.com/products/api-design/), signup and get an API key.

4. Replace the `X-RapidAPI-Key` in the `.env` file with your API key.

5. Finally, run the `main.py` file with the following command `python maim.py`.


You can change the stock symbol, interval, outputsize and active duration in the `main.py` file.

Note that not specifying an active duration will make the program run till the stock market is closed, if you choose to do this, endevour to subscribe to a plan that offers unlimited API credits (which the basic plan does not offer), failure to do so will result in an error after daily qouta of API credits have been used up.