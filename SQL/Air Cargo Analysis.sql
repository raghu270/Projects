create database aircargo;
use aircargo;

select * from customer;
select * from passengers_on_flights;
select * from routes;
select * from ticket_details;

/* Write a query to create route_details table using suitable data types for the fields, such as route_id, flight_num, 
origin_airport, destination_airport, aircraft d, and distance_miles. Implement the check constraint for the flight 
number and unique constraint for the route_id fields. Also, make sure that the distance miles field is greater than 0.*/

create table route_details(route_id int,flight_num int check(flight_num > 1000),
origin_airport varchar(225),destination_airport varchar(225),aircraft_id varchar(225),
distance_miles int check(distance_miles >0));

/* Write a query to display all the passengers (customers) who have travelled in routes 01 to 25. Take data from the 
passengers_on_flights table. */

select first_name from customer c inner join passengers_on_flights p 
on c.customer_id=P.customer_id where route_id between 1 and 25;

/*Query to identity the number of passengers and total revenue in business class from the ticket_details table*/

select COUNT(customer_id) as Number_of_Passengers,
sum(Price_per_ticket) as total_revenue from ticket_details where class_id='Bussiness';

/* Query to display the Full name ot the customer by extracting the first name and last name trom the customer table.*/

select CONCAT(first_name,' ' ,last_name) as full_name from customer ;

/*  Query to extract the customers who have registered and booked a ticket. use data from the customer and 
ticket_details tables */

select distinct t.customer_id from ticket_details t inner join customer c on t.customer_id=c.customer_id;

/* Query to identity the customer's first name and last name based on their customer ID and brand (Emirates) from the 
ticket_details table.*/

select distinct concat(first_name,' ' ,last_name) as full_name  from customer c inner join ticket_details t
on c.customer_id=t.customer_id where brand= 'Emirates';

/*Query to identity the customers who have travelled by Economy Plus class 
using Group By and Having clause on the passenger_on_flight  table*/

select COUNT(customer_id) as total_customers from passengers_on_flights 
group by class_id having class_id='Economy Plus';

/* Query to identity whether the revenue has crossed 10000 using the IF 
clause on the ticket_details table */

select IF(SUM(Price_per_ticket)>10000 ,'yes revenue has crossed 10000','no revenue has not crossed ')
from ticket_details;

/*Query to create and grant access to a new user to perform operations on a database*/

GRANT ALL ON *.* TO 'root'@'localhost';

/*Query to find the maximum ticket price for each class using window functions on the ticket_details table*/

select * from ticket_details
select class_id ,MAX(Price_per_ticket) over (partition by class_id) as max_price from ticket_details;

/*Query to extract the passengers whose route ID is 4 by improving the speed and performance of the passenger_on_flight table*/
SELECT customer_id FROM `passengers_on_flights_csv` WHERE route_id=4;

/*For the route ID 4, write a query to view the execution plan of the passengers_on_flight table*/
SELECT * FROM `passengers_on_flights_csv` WHERE route_id=4;

/*A query to calculate the total price of all tickets booked by a customer across different aircraft IDs using rollup function*/
SELECT customer_id,aircraft_id,SUM(Price_per_ticket)AS Total_sales FROM ticket_details_csv GROUP BY customer_id,aircraft_id WITH ROLLUP;

/*Query to create a view with only business class customers along with the brand of airlines*/
CREATE VIEW Bussiness_Class AS
SELECT customer_id,brand FROM `ticket_details_csv` WHERE class_id='Bussiness';
SELECT * FROM Bussiness_Class;












