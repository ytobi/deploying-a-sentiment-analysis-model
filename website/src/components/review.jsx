import React, { Component } from 'react'

class Review extends React.Component {

    render() { 
        const {id, text, sentiment} = this.props.review;
        return  (
            <div className="conatiner col mt-2" style={this.getReviewStyle()} >
                <div className="p-0 row d-flex flex-row" style={{backgroundColor: "lightgray"}}>
                    <span className='align-self-center m-2' style={sentiment >= 0.5 ? {fontWeight: "bold", color: "green"}: {fontWeight: "bold", color: "red"}}> 
                            { sentiment >= 0.5 ? "POSITIVE" : "NAGATIVE" }
                    </span>
                    <button onClick={() => this.props.onDelete(id)} className='btn btn-danger btn-md m-2 ml-auto'>Delete</button>
                </div>
                <div className="" style={{fontSize: 18}}>
                    {text}
                </div>
            </div>
        );
    }

    getReviewStyle() {
        return this.props.review.sentiment >= 0.5 ? {border: "solid 1px green"} : {border: "solid 1px red"}
    }
}
 
export default Review;