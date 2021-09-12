
# Labelling new music into playlists with CNN model

I usually put my musics into 2 playlists, a 'high' that wakes me up, is more upbeat, and is something I'm happy to listen to in the morning. A 'slow' list, which is something I'd like to hear at night. The music I listened to are mostly hip-hop, so it's not as easy as hip-hop = 'high', folk = 'slow' type of thing.

Spotify would be great for this, it has music features like energy, danceability, and valence, among other useful features. However, some of my musics aren't available on spotify so I've been purchasing single songs and manage it myself. The usual process involves me listening to new songs 1 by 1, deciding which list it belongs to.

Now I've had almost 400 songs in my phone, I built a ML model that would learn my taste/way to classify songs to save me time.

I started my exploratory work with common classification techniques, along with PCA, then moved onto neural networks with structured data (see exploratory). However, none of the result test accurarcy were good enough (roughly 70%-ish, my target was ~80% given the binary classication). It was only when I change the input time to perserve the time element in my songs that boost the accuracy by 10%, finally hitting the 80% goal. It was an example of how data in was more important than the different algorithms used.

Dataset under /data are omitted due to the large size, placeholder files are used in this repo.

## Acknowledgements

 - [Notes on Music Information Retrieval](https://musicinformationretrieval.com/index.html)
 - [Sound Feature Extraction](https://maelfabien.github.io/machinelearning/Speech9/#)
 - [The dummy’s guide to MFCC](https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd)
 - [Structured data learning with Wide, Deep, and Cross networks](https://keras.io/examples/structured_data/wide_deep_cross_networks/)
 - [Timeseries classification from scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/)

  
## Badges

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

  
