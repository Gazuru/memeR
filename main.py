from __future__ import print_function

import PIL
from PIL import Image, ImageOps

import tqdm

import numpy as np

import glob

import tensorflow as tf

import keras
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential, load_model
from keras import backend as K

train_dir = "C:/Users/energ/Desktop/témalab/train/train"
valid_dir = "C:/Users/energ/Desktop/témalab/valid/valid"


# print(train_dir + valid_dir)


def load_images():
    for i in range(num_of_classes):
        for f in glob.iglob(train_dir + "/" + class_names[i] + "/*"):
            images.append(Image.open(f))


def load_valid():
    for i in range(num_of_classes):
        for f in glob.iglob(valid_dir + "/" + class_names[i] + "/*"):
            val_images.append(Image.open(f))


def max_res():
    max_width, max_height = 0, 0
    for i in range(len(images)):
        if images[i].size[0] > max_width and images[i].size[1] > max_height:
            max_width, max_height = images[i].size
    return max_width, max_height


def pad_images(x):
    for i in range(len(x)):
        im_width, im_height = x[i].size
        delta_w = _WIDTH - im_width
        delta_h = _HEIGHT - im_height
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        x[i] = ImageOps.expand(x[i], padding)


def open_largest():
    for i in range(len(images)):
        if images[i].size == (_WIDTH, _HEIGHT):
            images[i].show()


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

images = []
val_images = []

load_images()
load_valid()

_WIDTH, _HEIGHT = max_res()

pad_images(images)
pad_images(val_images)


# összes kép nagy tömbben, shuffle, és beadni train
# annyi neuron az utolsó rétegen, ahány osztály lehet a kimenet
# teszt először kevesebb osztállyal
# HOT_ENCODING - az osztályok enkódolása
"""
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# ez a rész fent lényegkiemelés

model.add(Flatten())

# ez összeköti a fully connected részt a lényegkiemeléssel

model.add(Dense(64, activation='relu'))
model.add(Dense(num_of_classes))

# fully connected

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit()

model.summary()
"""