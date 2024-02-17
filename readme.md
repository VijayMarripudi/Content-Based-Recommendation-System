# Content Based Recommendation System
# Movie Recommendation System

## Overview

This project implements a movie recommendation system using a combination of content-based filtering and collaborative filtering techniques. It consists of three main components:

1. **Web Interface**: Provides a user-friendly interface for users to input their preferences and view movie recommendations.

2. **ML Model**: Utilizes machine learning algorithms to generate movie recommendations based on user input and movie features.

3. **Flask Code**: Integrates the web interface with the ML model to provide a seamless user experience.

## Implementation Details

### Web Interface

The web interface is built using HTML and CSS, with two main pages:

- `index.html`: Displays movie recommendations based on user input.
- `after.html`: Shows movie recommendations after processing user input.
- `error.html`: Displays an error message if user input is incorrect.

### ML Model

The ML model is responsible for generating movie recommendations. It consists of several steps:

1. Data Preprocessing: Cleans and preprocesses movie data.
2. Content-Based Filtering: Uses TF-IDF and cosine similarity to recommend movies based on similarity of features.
3. Collaborative Filtering: Implements SVD-based collaborative filtering for personalized recommendations.

### Flask Code

The Flask application acts as the middleware between the web interface and the ML model. It handles user input, passes it to the ML model for processing, and displays the recommendations on the web interface.

- `app.py`: Contains Flask routes for rendering web pages and processing user input.
- `templates/`: Directory containing HTML templates for the web interface.


## Modules need to be install to run the code.

- Flask <br>
cmd: pip install flask <br>
- Pandas <br>
cmd: pip install pandas <br>
- Numpy <br>
cmd: pip install numpy <br>
- Scikit-Learn <br>
cmd: pip install scikit-learn <br>
- Surprise <br>
 pip install scikit-surprise <br>
If above command is not working we need to change our environment to Anaconda <br>
In anaconda install above modules upto surprise<br>
For scikit surprise use below command  <br>
conda install -c conda-forge scikit-surprise <br>


## Usage

To run the application locally:

1. Clone the repository: `git clone <repository-url>`
2. Navigate to the project directory: `cd movie-recommendation-system`
3. Install modules that listed above
4. Run the Flask application: `python app.py`
5. Access the web interface in your browser at `http://127.0.0.1:5000/`