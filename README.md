# Movie Recommendation

Film recommendation system based on previous ratings.
## Description
The program uses the **k-nearest neighbors** algorithm to identify the most relevant videos based on user ratings. To calculate the distance, and actually to calculate the similarity of individual vectors (films with ratings as elements of the vector), the **cosine similarity** was used.
### Built with
* **Python 3** (Numpy, Pandas)
## Usage
Enter the title of the movie you want to find recommendations for in **line 78** `name_of_movie`.
## Dataset
A collection hosted on [Kaggle](https://kaggle.com/shubhammehta21/movie-lens-small-latest-dataset/) from the MovieLens website was used, which ultimately deals with recommending movies for users. The dataset contains over 100 000 ratings and 3686 tags for 9742 movies.

At the outset, the amount of data was reduced to improve the operation of the system. Users were randomly selected and have rated at least 50 videos remain in the collection. Videos must have at least 10 ratings. NaN values for no evaluation were changed to more intuitive zeros.

