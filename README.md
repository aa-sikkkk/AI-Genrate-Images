AI Image Generator
This is a project that generates images using a pre-trained GAN (Generative Adversarial Network) model. The user inputs a sentence and the model generates an image based on that sentence.

Getting Started
To get started with this project, you'll need to install Python 3 and a few libraries:

torch
torchvision
python-telegram-bot
You can install these libraries using pip:


pip install torch torchvision python-telegram-bot
Once you have the libraries installed, you can run the program using the following command:


python ImageGeneratorAI.py
Troubleshooting
If you encounter any problems while running this program, feel free to contact me at AhmeedSheeko@gmail.com.

One common issue that has been reported is a traceback error that looks like this:

Traceback (most recent call last):
   File "C:\Users\PC\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\telegram\ext\dispatcher.py", line 555, in process_update
     handler. handle_update(update, self, check, context)
   File "C:\Users\PC\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\telegram\ext\handler.py", line 198, in handle_update
     return self. callback(update, context)
   File "e:\Projects\AI Genrate Images\ImageGeneratorAI.py", line 154, in <lambda>
     start_handler = CommandHandler('start', lambda update, context: start(update, context, my_generator), pass_args=True, pass_job_queue=True, pass_chat_data=True)
   File "e:\Projects\AI Genrate Images\ImageGeneratorAI.py", line 53, in start
     generate_image(update, context, my_generator)
   File "e:\Projects\AI Genrate Images\ImageGeneratorAI.py", line 87, in generate_image
     latent_code = get_latent_code(input_text, my_generator)
   File "e:\Projects\AI Genrate Images\ImageGeneratorAI.py", line 119, in get_latent_code
     text_tensor = torch. Tensor([input_text])
ValueError: too many dimensions 'str'

If you encounter this error, it's likely because the input text has too many dimensions. You can try converting the input text to a 1-dimensional tensor like this:

scss

text_tensor = torch.Tensor([input_text]).squeeze()
I hope this helps, and happy generating!
