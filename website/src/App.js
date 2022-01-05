import logo from "./logo.svg";
import NavBar from "./components/navbar";
import "./App.css";
import React, { Component } from "react";
import InputComponent from "./components/inputComponent";
import Review from "./components/review";
class App extends React.Component {
  state = {

    reviews: [
    ]
  };

  constructor() {
    super();
  }

  handleDelete = (reviewId) => {
    const reviews = this.state.reviews.filter((r) => r.id !== reviewId);
    this.setState({ reviews });
  };
  handleAddReview = (data) => {
    let reviews = [...this.state.reviews];
    reviews = [{id: this.state.reviews.length, text: data.review, sentiment: Number(data.sentiment)}].concat(reviews);
    this.setState({reviews});
  };
  render() {
    return (
      <React.Fragment>
        <NavBar
          totalCounters={8}
        />
        <main className="container">
          <InputComponent
          onAddReview={this.handleAddReview}
          ></InputComponent>
          
          <div className="conatiner p-3" style={{maxHeight: "50vh", minHeight: "50vh", overflowY: "auto", overflowX: "hidden"}}>

            {this.state.reviews.map((review) => (
              <Review
              key={review.id}
              review={review}
              onDelete={this.handleDelete}
              ></Review>
           ))}
          </div>
        </main>
        <div className="d-flex flex-row justify-content-center" style={{maxHeight: "47px", fontWeight: "small"}}>
          <span>Built with love by</span>
          <a className="ml-1" href="https://ytobi.github.io/" style={{ color: "" }} rel="noreferrer" target={"_blank"}>Tobi Obadiah</a>
        </div>
      </React.Fragment>
    );
  }
}

export default App;
