import React from "react";
import botpic from "../images/botpic.jpeg";

const NavBar = ({ totalCounters }) => {
  return (
    <nav className="navbar " style={{ backgroundColor: "#0177ff", color: "white" }}>
      <div className="container d-flex justify-content-center">
          <h2 className="container d-flex justify-content-center">
            <img className="" src={botpic} alt="botpic" height={50} />
            <span> Sentiment Analyzer For Movie Reviews</span>
              {" "}
          </h2>
           <div className="container  d-flex justify-content-center">
              Type a review and click on the submit button to examine its sentiment.
          </div>
          <div className="container d-flex justify-content-center">
            You can find sample reviews 
            <a className="m-1" href="https://evolutionwriters.com/samples_and_examples/movie_reviews/" style={{ color: "gold" }} rel="noreferrer" target={"_blank"}>here</a>
             or 
            <a className="m-1" href="https://academichelp.net/samples/academics/reviews/movie/" style={{ color: "gold" }} rel="noreferrer" target={"_blank"}>here.</a>
          </div>
      </div>
    </nav>
  );
};

export default NavBar;
