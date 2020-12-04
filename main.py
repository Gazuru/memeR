from __future__ import print_function

import glob
import random

import numpy as np
from PIL import Image, ImageOps
from keras.layers import (Conv2D, Dense, Flatten, MaxPooling2D)
from keras.models import Sequential
from sklearn.utils import shuffle
import tensorflow
from matplotlib import pyplot as plt

train_dir = "D:/BME/5/Témalabor/train/train"
valid_dir = "D:/BME/5/Témalabor/valid/valid"


# print(train_dir + valid_dir)

def load_images():
    for i in range(num_of_classes):
        for f in glob.iglob(train_dir + "/" + class_names[i] + "/*"):
            images.append(Image.open(f).convert('RGB'))
            labels.append(i)


def load_valid():
    for i in range(num_of_classes):
        for f in glob.iglob(valid_dir + "/" + class_names[i] + "/*"):
            val_images.append(Image.open(f).convert('RGB'))
            val_labels.append(i)


def max_res():
    max_width, max_height = 0, 0
    for i in range(len(images)):
        if images[i].size[0] > max_width and images[i].size[1] > max_height:
            max_width, max_height = images[i].size
    return max_width, max_height


def downsample_images(im):
    for i in range(len(im)):
        im_width, im_height = im[i].size
        if im_width == im_height:
            im[i] = im[i].resize(size, Image.ANTIALIAS)
        elif im_width > im_height:
            ratio = size[0] / im_width
            ns = (int(im_width * ratio), int(im_height * ratio))
            im[i] = im[i].resize(ns, Image.ANTIALIAS)
        else:
            ratio = size[1] / im_height
            ns = (int(im_width * ratio), int(im_height * ratio))
            im[i] = im[i].resize(ns, Image.ANTIALIAS)


def pad_images(im):
    for i in range(len(im)):
        if im[i].size < size:
            im_width, im_height = im[i].size
            delta_w = size[0] - im_width
            delta_h = size[1] - im_height
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            im[i] = ImageOps.expand(im[i], padding)


class_names = ["10_guy", "advice_mallard", "afraid_to_ask_andy", "angry_advice_mallard", "archer",
               "awkward_moment_sealion", "baby_insanity_wolf", "bad_luck_brian", "birth_control_effectiveness",
               "blue_button", "boardroom_meeting_suggestion", "but_thats_none_of_my_business", "car_salesman",
               "change_my_mind", "chopper_argument", "coma_patient", "confession_bear", "connor_playstation_button",
               "disappointed_black_guy", "distracted_boyfriend", "double_d_fact_book", "doubt_x", "drake_pref",
               "drew_scanlon_reaction", "expanding_brain", "fallout_wtf", "fear_no_man", "first_world_problems",
               "free_real_estate", "futurama_fry", "good_guy_greg", "gru_flipchart", "guess_ill_die", "hackerman",
               "hannibal_wack", "happy_seal", "hard_to_swallow_pills", "headache", "how_old_is", "i_guarantee_it",
               "i_killed_a_man", "i_miss_the_internet", "i_prefer_the_real", "incredibles_got_time",
               "intelligent_students_crying_kid", "introduce_ourselves", "is_this_a_pigeon", "jerry_pie_face",
               "jerry_shotgun", "jim_office_smiling", "joke_to_you", "keep_your_secrets", "knights_swords_in",
               "kowalski_analysis", "left_exit_12_off_ramp", "lets_see_who_this_really_is", "level_stress_99",
               "lisa_presentation", "lobsters_die", "mario_bros_views", "masters_blessing", "math_lady",
               "matrix_morpheus", "migrane_etc", "monkey_puppet_lookaway", "mugatu_so_hot_right_now",
               "of_course_i_know_him_hes_me", "one_does_not_simply", "other_women", "painting_sneaky", "patrick_wallet",
               "paul_ryan_screen", "people_with_signs", "pepperidge_farm_remembers", "peter_griffin_news",
               "philosoraptor", "picard_wtf", "pikachu_o", "predator_epic_handshake", "quiz_kid", "ralph_in_dangee",
               "rewind_time", "rollsafe_headtap", "sacred_texts", "savage_patrick", "say_the_line_bart",
               "scientist_myself", "scumbag_boss", "scumbag_steve", "sisters_name", "sleeping_shaq",
               "so_i_got_that_goin_for_me_which_is_nice", "socially_awesome_awkward_penguin", "spiderman_double_point",
               "squidward_looking", "stefan_pref", "steve_harvey_conflicted", "success_kid", "sudden_clarity_clarence",
               "teresea_may_dance", "thanos_strongwill", "that_wasnt_very_cash_money", "that_would_be_great",
               "the_most_interesting_man_in_the_world", "the_scroll_of_truth", "this_is_brilliant_but_i_like_this",
               "thor_defeat", "time_for_crusade", "tom_jerry_paper", "too_damn_high", "two_red_buttons", "tyler_lie",
               "we_dont_do_that_here", "wednesday_frog", "who_killed_hannibal"]

num_of_classes = 10
size = 100, 100

images = []
labels = []
val_images = []
val_labels = []

load_images()
load_valid()
labels = np.array(labels)
val_labels = np.array(val_labels)

downsample_images(images)
downsample_images(val_images)

pad_images(images)
pad_images(val_images)

for i in range(len(images)):
    images[i] = np.array(images[i].getdata()).reshape(size[1], size[0], 3)

images = np.asarray(images)

for y in range(len(val_images)):
    val_images[y] = np.array(val_images[y].getdata()).reshape(size[1], size[0], 3)

val_images = np.asarray(val_images)

# összes kép nagy tömbben, shuffle, és beadni train

images, labels = shuffle(images, labels, random_state=25)
images = images / 255.0
val_images = val_images / 255.0

# annyi neuron az utolsó rétegen, ahány osztály lehet a kimenet
# teszt először kevesebb osztállyal
# HOT_ENCODING - az osztályok enkódolása

n_train = images.shape[0]
n_test = val_images.shape[0]

print("Number of training examples: {}".format(n_train))
print("Number of testing examples: {}".format(n_test))

tensorflow.config.experimental.list_physical_devices('GPU')

model = Sequential()
model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(size[1], size[0], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# ez a rész fent lényegkiemelés

model.add(Flatten())

# ez összeköti a fully connected részt a lényegkiemeléssel

model.add(Dense(64, activation='relu'))
model.add(Dense(num_of_classes, activation='softmax'))

# fully connected

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(images, labels, batch_size=128, epochs=18, validation_split=0.3)

test_loss = model.evaluate(val_images, val_labels)

rnd_test_num = 10

for i in range(rnd_test_num):
    test_image = val_images[random.randrange(num_of_classes * 40)]

    pred = model.predict(np.array([test_image]))

    plt.imshow(test_image, interpolation='nearest')
    plt.title(class_names[np.argmax(pred)])
    plt.show()
