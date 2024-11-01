create database music;
use  music;

select * from artist;
select * from album;
alter table album add foreign key(artist_id) references artist(artist_id);
select * from playlist;
select * from playlist_track;
alter table playlist_track add foreign key(playlist_id) 
references playlist(playlist_id);
select * from employee;
select * from customer;
select * from track;
alter table track add foreign key(track_id)
references playlist_track(track_id);
alter table track add foreign key(album_id)
references album(album_id);
select * from media_type;
select * from genre;
select * from invoice;
alter table invoice add foreign key(customer_id)
references customer(customer_id);
select * from invoice_line;
alter table invoice_line add foreign key(invoice_id)
references invoice(invoice_id);
alter table invoice_line add foreign key(track_id)
references track(track_id);

/*	Question Set 1 - Easy */

/* Q1: Who is the senior most employee based on job title? */

select employee_id,last_name,first_name,title,
from employee order by levels desc limit 1

select employee_id,last_name,first_name,title
from employee where levels=(select max(levels) from employee);

/* Q2: Which countries have the most Invoices? */

select billing_country ,count(invoice_id) as no_of_invoices from invoice group by billing_country
order by no_of_invoices desc;

/* Q3: What are top 3 values of total invoice? */

select total from invoice
order by total desc limit 3; 

/* Q4: Which city has the best customers? We would like to throw a promotional Music Festival in the city we made the most money. 
Write a query that returns one city that has the highest sum of invoice totals. 
Return both the city name & sum of all invoice totals */

select billing_city,sum(total) as total_city_wise_invoice from invoice 
group by billing_city order by total_city_wise_invoice desc limit 1;

/* Q5: Who is the best customer? The customer who has spent the most money will be declared the best customer. 
Write a query that returns the person who has spent the most money.*/

select c.customer_id,first_name,last_name ,sum(total) as totall from 
customer c join invoice i
on c.customer_id=i.customer_id group by c.customer_id ,first_name,last_name
order by totall desc limit 1

/* Question Set 2 - Moderate */

/* Q1: Write query to return the email, first name, last name, & Genre of all Rock Music listeners. 
Return your list ordered alphabetically by email starting with A. */



select distinct email,first_name,last_name 
from customer c join invoice i on
c.customer_id=i.customer_id join invoice_line il on 
i.invoice_id=il.invoice_id join track t on
il.track_id=t.track_id join genre g
on
t.genre_id=g.genre_id where g.name='Rock'
order by email




