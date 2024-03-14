import streamlit as st
import pickle as pkl

st.title('Book Recommendation( Group 5) ')
df = pkl.load(open(r"C:\\Users\\Gopinath\\df.pkl", "rb"))
model = pkl.load(open(r"C:\\Users\\Gopinath\\model.pkl", "rb"))
data = pkl.load(open(r"C:\\Users\\Gopinath\\data.pkl", "rb"))


def recommend(books):
    recommended_book_names = []
    distances, indices = model.kneighbors(df.loc[books].values.reshape(1, -1), n_neighbors=10)
    print("\nRecommended books:\n")
    for i in range(0, len(distances.flatten())):
        if i > 0:
            recommended_book_names.append(df.index[indices.flatten()[i]])
    return recommended_book_names


users = data['User-ID'].values
Users_book = st.selectbox('Select a User-ID from drop down', users)
selected_user_id = data[data["User-ID"] == Users_book].sort_values('Book-Rating', ascending=False).head(1)
selected_book = selected_user_id['Book-Title'].values

st.write('You selected:', Users_book)


if st.button('Show Recommend book'):
    recommended_book = recommend(selected_book)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(recommended_book[0])

    #with col2:
        st.markdown(recommended_book[1])

    #with col3:
        st.markdown(recommended_book[2])

    #with col4:
        st.markdown(recommended_book[3])

    #with col5:
        st.markdown(recommended_book[4])
 # Use the full page instead of a narrow central column
    #st.set_page_config(layout="wide")

    # Space out the maps so the first one is 2x the size of the other three
    #col1, col2, col3, col4,col5 = st.columns((2, 1, 1, 1,1))


