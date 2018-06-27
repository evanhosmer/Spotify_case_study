import pandas as pd
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
sp = spotipy.Spotify()
cid = ""
secret = ""
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace=False

def get_songs(pl_ids):
    for uri in pl_ids:

       results = sp.user_playlist_tracks(uri.split(':')[2], uri.split(':')[4])
       tracks = results['items']

       # Loops to ensure I get every track of the playlist
       while results['next']:
           results = sp.next(results)
           tracks.extend(results['items'])

    return tracks

def put_in_df(x):
    dffinal = pd.DataFrame()
    for i in range(len(x)):
        data = pd.DataFrame(x[i])
        dffinal = dffinal.append(data)
    return dffinal

def popularity(df):
    pops = [df[i]['track']['popularity'] for i in range(len(df))]
    return pops

def name_and_artist(df):
    new = []
    for i in range(len(df)):
        artist = df[i]['track']['artists'][0]['name']
        songtitle = df[i]['track']['name']
        songlength = df[i]['track']['duration_ms'] / 60000
        new.append([artist,songtitle,songlength])
    return new

def get_playlist_data(playlist):
    songs = get_songs(playlist)
    audio = [sp.audio_features(songs[i]['track']['id']) for i in range(len(songs))]
    final = put_in_df(audio)
    pops = popularity(songs)
    se = pd.Series(pops)
    final['Popularity'] = se.values
# return songs, audio, final, pops
    data = name_and_artist(songs)
    data2 = pd.DataFrame(data,columns = ['Artist','Title','Duration(m)'])
    one = data2.reset_index()
    two = final.reset_index()
    one.pop('index')
    two.pop('index')
    finaldf = pd.concat([one,two],axis = 1)
    finaldf.drop(['type','duration_ms','analysis_url','id','uri'], axis=1,inplace = True)
    return finaldf

alternative = get_playlist_data(['spotify:user:evanhosmer:playlist:7bMVeYcq2l2B3j0ZDcRkFj'])
hard_rock = get_playlist_data(['spotify:user:evanhosmer:playlist:25XT3aF13nNS1TumGBEJ5i'])
metal_core = get_playlist_data(['spotify:user:evanhosmer:playlist:1fmpGfBh9LOolb3p6w6d7N'])
