import requests
import telegram
from telegram.ext import CommandHandler, MessageHandler, Filters, Updater
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import os
import datetime
from torch.utils.data import DataLoader
import torch.nn as nn

# Define the image folder path
IMAGE_FOLDER = 'images'

# Create the image folder if it doesn't exist
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Create a dataset from the images in the directory
dataset = ImageFolder(root=IMAGE_FOLDER, transform=transform)

# Create a data loader that loads images from the dataset in batches
batch_size = 64
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

batch_size = 32
shuffle = True

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Save the data loader to a file so that it can be used in the training script
torch.save(data_loader, 'data_loader.pth')

# Load the trained generator network
generator = torch.load('data_loader.pth')

# Replace YOUR_TOKEN with your bot's API token
bot = telegram.Bot(token='6121468637:AAHfUZy_1wGXN9W5ZxHo5P8gogMJY4X6o6c')

def start(update, context, my_generator):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Hello! Please describe the image you want me to generate. Be as specific as possible, such as the colors, objects, shapes, etc.")
    # if user sends a description, generate an image from it
    if not update.message.text:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Please describe the image you want me to generate. Be as specific as possible, such as the colors, objects, shapes, etc.")
    else:
        generate_image(update, context, my_generator)
        

def train(update, context):
    # Check if there is a saved image for the user
    user_image_path = os.path.join(IMAGE_FOLDER, str(update.effective_chat.id) + '.jpg')
    if not os.path.exists(user_image_path):
        context.bot.send_message(chat_id=update.effective_chat.id, text="Please send me an image to train the model.")
        return
    # Call the save_image function to save the image
    save_image(update, context)


def save_image(update, context):
    # Checl if the message contains an image
    if not update.message.photo:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Please send me an image.")
        return
    # Get the file ID of the image sent by the user
    file_id = update.message.photo[-1].file_id
    # Download the photo
    file = context.bot.get_file(file_id)
    # Name the image file with the user's ID
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-UTC")
    filename = f"{update.effective_chat.id}-{now}.jpg"
    # Save the image file
    file.download(os.path.join(IMAGE_FOLDER, filename))
    context.bot.send_message(chat_id=update.effective_chat.id, text="Image saved successfully!")    

def generate_image(update, context, my_generator):
    # Get the user's input from the message text
    input_text = update.message.text

    # Convert the user's input into a latent code
    latent_code = get_latent_code(input_text, my_generator)

    # Generate an image from the latent code
    with torch.no_grad():
        image = my_generator(latent_code)

    # Convert the image tensor to a PIL Image and save it
    transform = transforms.ToPILImage()
    image = transform(image)
    image.save('generated_image.jpg')

    # Send the generated image to the user
    with open('generated_image.jpg', 'rb') as f:
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=f)

def echo(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

def get_latent_code(input_text, my_generator):
    # Define the input size, hidden size, and latent size
    input_size = 100
    hidden_size = 256
    latent_size = 100

    # Define the text encoder network
    text_encoder = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, latent_size),
        nn.Tanh()
    )
    # Convert the input text to a tensor
    text_tensor = torch.Tensor([input_text])

    # Generate a latent code from the text tensor
    with torch.no_grad():
        latent_code = text_encoder(text_tensor)

    return latent_code


# command handler to get image from description
def generate_images(update, context):
    description = update.message.text.split('/generate')[1]
    r = requests.get(f'https://api.unsplash.com/photos/random?query={description}&client_id=xKiAshqqBCZJ1QfaXMhkvCLfT_qhvcZq64kVfzMzoX8')
    if r.status_code == 200:
        photo_url = r.json()['urls']['regular']
        bot.send_photo(chat_id=update.effective_chat.id, photo=photo_url)
    else:
        bot.send_message(chat_id=update.effective_chat.id, text='Sorry, could not find an image.')

# message handler to reply to non-command messages
def reply(update, context):
    bot.send_message(chat_id=update.effective_chat.id, text='Sorry, I did not understand that. Please use /generate to get an image from a description.')



def main():
    updater = Updater(token='6121468637:AAHfUZy_1wGXN9W5ZxHo5P8gogMJY4X6o6c', use_context=True)
    dispatcher = updater.dispatcher

    # Create a generator object
    #generator = MyGenerator()
    # Define my_generator
    my_generator = generator

    # Register the start() function with the generator argument
    start_handler = CommandHandler('start', lambda update, context: start(update, context, my_generator), pass_args=True, pass_job_queue=True, pass_chat_data=True)



    #start_handler = CommandHandler('start', start)
    generate_image_handler = MessageHandler(Filters.text & (~Filters.command), generate_image)
    train_handler = CommandHandler('train', train)
    image_handler = MessageHandler(Filters.photo & (~Filters.command), save_image)
    echo_handler = MessageHandler(Filters.text & (~Filters.command), echo)

    # add handlers to dispatcher
    dispatcher.add_handler(CommandHandler('generate', generate_images))
    dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), reply))
    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(generate_image_handler)
    dispatcher.add_handler(train_handler)
    dispatcher.add_handler(image_handler)
    dispatcher.add_handler(echo_handler)

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()