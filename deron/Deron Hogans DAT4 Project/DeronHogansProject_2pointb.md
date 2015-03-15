<h1>Question</h1>

<p>Can I predict a start up’s potential for venture capital or angel investment?</p>

<blockquote>
<p>Crunchbase provides in open data map (CSV) of start-ups that includes information like the name of the organization, location and social presence. Crunchbase also offers API to access information more relevant to answering my question:
*Valuation
*IPO (if any)
*Fundraise
*Acquisition</p>
</blockquote>

<p>There is no Python wrapper on Crunchbase’s site. 
I’m interested in learning more about how VC’s and angel investors identify value in start-ups.</p>

<h1>Project Pivot: 2pointb Trip Data</h1> 

<p>After a bit of frustration and some disappointment, I've decided to change direction with my data science project and focus on something a bit more useful and scalable. As the digital marketing strategist for ride sharing tool, 2pointb, I felt using some of their available data would have more utility and be better suited for the purposes of this project.</p>


<h1>Predicting Applied Discounts Using Python - 1st Draft</h1> 

<p>2pointb is a fairly new player in the ridesharing market and is aggressively soliciting users. Promotional codes for dicsounts on rides are used frequently as a marketing tool. Currently, there is no insight to inform strategic focus on new or existing users when spreading the word about promotional codes. Ride discount values are not informed by actual use. The aim of this exploration is to identify correlated indicators for discount rates and potentially use those indicators to predict discount rates.</p>

<h2>Data Gathering</h2> 

<p>The 2pointb app produces a data set of ride metrics. I downloaded this file as a CSV directly from 2pointb's metrics reporting suite. The data set includes the following information: 
*Rider ID
*Rider Name
*Ride Status
*Driver
*Fleet
*Vehicale Make and Model
*Gross Ride Amount
*Promo Code
*Promo Code Used
*Discount 
*And some other information

After a brief review of the data, I noticed most of the information was either for specific identification or categorical (Rider Name, Vehicle Make and Model, etc). I decided to keep Fleet (SUV, Cab, Discount Cab) out of a hunch that this may be a Discount Rate predictor. I opted to remove this information from the DataSet manually to save time for this first draft. After removing unwanted data, I read this CSV into a python file as a DataFrame to initiate exploration</p>

<h2>Exploration</h2> 

<p>I continued transforming the data in preparation for model evaluations and testing. I transformed trip status to numerical values ('Meter Off: 1', 'Unable to Auth': 0, 'Cancelled': -1). I then transformed Trip Adjustments and Existing Passenger to numerical values (Yes: 1, No: 0). Lastly, I transformed Fleet to numerical values ('Discount Cab': 1, 'Cab': 2, 'Black': 3, 'Ridehare': 4, 'SUV': 5).</p>

<p>After visualizing a few scatter plots of various features and their relation to Discount Rates, I noticed a bit of correlation between the following features and Dicsount Rates:
*Existing Passenger
*Gross Amount
*Fleet</p>

<p>I selected these features and began building a model evaluation using K N-Neighbors. I received a KNN score of .77 at K = 5, but a K_range estimator would reveal 9 as the optimal value of K. The K value of 9 would prove to be more "accurate" but further tests of sensitivity and specificity are needed to truly identify the best value for K.</p>

<h2>Next Steps</h2> 

<p>My next steps are to complete the construction of a well-fitted prediction model. I also plan to strengthen my data set and make it a bit more robust with data from the month of February (the current data set only reflects January data).</p>

<h2>Second Iteration</h2> 

<p>After testing my initial model on January trips, I added February trips to the data set and proceeded to implement my model predicting trip discounts with the following features: 
*Existing Passenger
*Gross Amount
*Fleet

At K=5, my KNN prediction model socred 79% metrics accuracy score. I also employed a LinearSVC to see what the best predictor would be and Existing Passenger feature seems to be the best predictor for Discounts</p>

