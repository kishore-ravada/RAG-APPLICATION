# Database Management Systems & SQL

## Relational Databases & SQL JOINs
A Relational Database Management System (RDBMS) stores data in tables linked by foreign keys.

### INNER JOIN
`INNER JOIN` selects records that have matching values in both tables. If a student has no corresponding enrollment record, they will NOT appear in the inner join result.

```sql
SELECT Students.name, Courses.course_name
FROM Students
INNER JOIN Enrolments ON Students.id = Enrolments.student_id;
```

### LEFT JOIN (LEFT OUTER JOIN)
`LEFT JOIN` returns ALL records from the left table, and matched records from the right table. If there is no match, NULL values appear on the right side.

Common misconception: Confusing `INNER JOIN` with `LEFT JOIN`. Students often use `INNER JOIN` when they need to retain all records from the primary table regardless of match status.

## Primary Keys vs Foreign Keys
- Primary Key: Uniquely identifies each record in a table (cannot be NULL).
- Foreign Key: A field in one table that refers to the Primary Key in another table to enforce referential integrity.
