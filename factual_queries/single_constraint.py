import pickle
import numpy as np

def load_basketball_players():
    with open("./factual_queries/data/basketball_players.pkl", "rb") as f:
        items = pickle.load(f)

    prompt_template = "Tell me the year the basketball player {} was born in."
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: The player was born in"
    for item in items:
        item["constraint"] = item["player_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["constraint"]))
        item["label"] = item["birth_year"]
        item["popularity"] = item["popularity"]
    return items


def load_football_teams():
    with open("./factual_queries/data/football_teams.pkl", "rb") as f:
        items = pickle.load(f)

    prompt_template = "Tell me the year the football team {} was founded in."
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: The team was founded in"
    for item in items:
        item["constraint"] = item["team_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["constraint"]))
        item["label"] = item["founding_year"]
        item["popularity"] = item["popularity"]
    return items


def load_songs():
    with open("./factual_queries/data/songs.pkl", "rb") as f:
        items = pickle.load(f)

    prompt_template = "Tell me the performer of the song {}"
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: The performer is"
    for item in items:
        item["constraint"] = item["song_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["constraint"]))
        item["label"] = item["artist_name"]
        item["popularity"] = item["popularity"]
    return items


def load_movies():
    with open("./factual_queries/data/movies.pkl", "rb") as f:
        items = pickle.load(f)
    prompt_template = "Tell me the director of the movie {}."
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: The director is"
    for item in items:
        item["constraint"] = item["movie_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["constraint"]))
        item["label"] = item["director_name"]
    return items

def load_counterfact_subset(subset):
    filename = f"./factual_queries/data/{subset}.pkl"
    items = pickle.load(open(filename, "rb"))
    return items