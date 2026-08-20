handle:
    what happens when the model return emptyx[x] (it stop the generation)
    if packages didn't exist
    if there no function name it's not count as a function
    if there is multiple keys or duplicates
    if the file is empty
    if one of the value is empty like (prompt is empty)

  
you should know:
    argparse
    json
    pydantic
    unicode
    why this structure of files why everything in src/ 

test:
    i/o errors:
            file not found
            directory
            permissions
    keyboardinterrupt add also exception

# Tasks:
check:
- .gitignore
- readme
- flake8
- mypy
- you should remove get_vocab() and decode_tokens (check if the vocab file didn't exist)
- remove the debugging informations
# recoding
- you should be able answer this "there is 1 blue car 3  apples green" to output this 
  ""car": \[1, "blue"], "apple": \[3, green] it's not necessary to give you function(apples is not wrong)



  # Completed:
    the trie doesn't work correctly it stops if you found "cat" and 
      don't care about "catfish" \[x]
