uv run python -m moulinette prepare_exercises --set private
cp -r data/input/* ../data/input/
cd ../
make run
cd moulinette
uv run python -m moulinette grade-student-answers ../data/output/function_calling_results.json --set private
