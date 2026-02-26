#!/bin/bash
set -x
sed -i 's/course id/course_id/g' dbms_concept.sql
sed -i 's/dept name/dept_name/g' dbms_concept.sql
sed -i '/^$/d' dbms_concept.sql
sed -i 's/tot salary/tot_salary/g' dbms_concept.sql
sed -i 's/max budget/max_budget/g' dbms_concept.sql

sed -i 's/dept total avg/dept_total_avg/g' dbms_concept.sql
sed -i 's/dept total/dept_total/g' dbms_concept.sql
sed -i 's/num instructors/num_instructors/g' dbms_concept.sql
sed -i 's/dept count/dept_count/g' dbms_concept.sql
sed -i 's/d count/d_count/g' dbms_concept.sql
sed -i 's/count proc/count_proc/g' dbms_concept.sql
sed -i 's/time slot id/time_slot_id/g' dbms_concept.sql
sed -i 's/time slot/time_slot/g' dbms_concept.sql
sed -i 's/s rank/s_rank/g' dbms_concept.sql
sed -i 's/student grades/student_grades/g' dbms_concept.sql
sed -i 's/dept rank/dept_rank/g' dbms_concept.sql
sed -i 's/dept grades/dept_grades/g' dbms_concept.sql
sed -i 's/num credits/num_credits/g' dbms_concept.sql
sed -i 's/tot credits/tot_credits/g' dbms_concept.sql
sed -i 's/avg total credits/avg_total_credits/g' dbms_concept.sql

