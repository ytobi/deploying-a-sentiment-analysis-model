import React, { Component } from 'react'
import axios from "axios"

class InputComponent extends React.Component {
    state = {
       review: "",
       loading: false,
    };
    handleGetSentiment = () => {
        
        this.setState({loading: true});

        const url = "https://8ro3fp6q8d.execute-api.us-east-1.amazonaws.com/v1/analyse";
        const data = {
            "review": this.state.review,
        };
        axios.post(url, JSON.stringify(data)).then((response) => {
            this.props.onAddReview({review: response.data.body.review, sentiment: response.data.body.sentiment });
            this.setState({loading: false});


        }).catch(error => {
            console.log("Error getting sentiment reivew");
            this.setState({loading: false});
        });
        
    };
    render() { 
        return (
            <div className="container mt-2">
                <div className='form-group d-flex flex-column justify-content-center'>
                    <div className='mt-auto align-self-center d-flex flex-column justify-content-center' style={{position: "absolute", display: 'block' }}>
                        <div hidden={ this.state.loading ? false : true } className="spinner-border text-primary align-self-center" role="status" >
                            <span className="sr-only">Loading...</span>
                        </div>
                        <span hidden={ this.state.loading ? false : true } className='' >Analyzing review...</span>
                    </div>
                    <label>Review:</label>
                    <textarea disabled={ this.state.loading ? true : false } defaultValue={this.state.review} onChange={(e) => {this.setState({review: e.target.value}) }} className="form-control"  rows="5" id="review" placeholder='Type your review here.'></textarea>
                    <span>
                    </span>
                </div>
                <button disabled={ this.state.loading ? true : false } onClick={() => this.handleGetSentiment()} className="btn btn-primary" type="submit">Submit</button>
            </div>
        );
    }
}
                
export default InputComponent;