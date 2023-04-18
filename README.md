# ImageGeneratorAI

This is a Telegram bot that generates images from text descriptions using a pre-trained generative model. The bot is based on Python and PyTorch and uses the Telegram Bot API.

## Getting started

To use the bot, you need to have a Telegram account and install the Telegram app on your mobile device or desktop computer. Then, follow these steps:

1. Clone the repository or download the source code.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Create a new Telegram bot and obtain the API token from [BotFather](https://core.telegram.org/bots#6-botfather).
4. Set the API token as an environment variable by running `export TELEGRAM_API_TOKEN=<your token>` on Linux or `set TELEGRAM_API_TOKEN=<your token>` on Windows.
5. Start the bot by running `python ImageGeneratorAI.py`.

## Troubleshooting

If you encounter any problems while running the bot, please check the following:

* Make sure that you have installed all the required dependencies and configured the environment variables correctly.
* If you get an error message that says `ValueError: too many dimensions 'str'`, try modifying the `get_latent_code` function in `ImageGeneratorAI.py` as follows:

```python
def get_latent_code(input_text, generator):
    with torch.no_grad():
        text_tensor = torch.Tensor([input_text])
        return generator.encode(text_tensor.cuda())
        ```

If you still cannot solve the problem, please contact the developer at AhmeedSheeko@gmail.com.

## Acknowledgements
The generative model used in this project is a pre-trained version of the StyleGAN2-ADA model developed by NVIDIA. The Telegram Bot API is provided by Telegram. Special thanks to OpenAI for providing the GPT-3.5 architecture for training the ChatGPT language model used to develop this bot.
