#!/usr/bin/env bash
#
# Create a new Jekyll post with the given title.

# --- Help Information ---
help() {
  echo "Usage:"
  echo "   bash ./new_post.sh \"Your Post Title Here\" [options]"
  echo
  echo "Example:"
  echo "   bash ./new_post.sh \"getting-started-with-llms\""
  echo
  echo "Options:"
  echo "     -h, --help           Print this help information."
}

# --- Argument Parsing ---
# The loop handles options like -h. Positional arguments (the title) are handled after.
while (($#)); do
  opt="$1"
  case $opt in
  -h | --help)
    help
    exit 0
    ;;
  # This catches any other options (e.g., -p, --unknown) and flags them as errors.
  -*)
    echo -e "> Error: Unknown option '$opt'\n"
    help
    exit 1
    ;;
  # If the argument doesn't start with '-', it's not an option.
  # We break the loop, assuming it's the post title.
  *)
    break
    ;;
  esac
done

# --- Main Logic ---
# After the loop, if $1 is empty, it means no post title was provided.
if [ -z "$1" ]; then
  echo -e "> Error: Post title is required.\n"
  help
  exit 1
fi

# The first positional argument is our post title.
# We wrap it in quotes to handle titles with spaces.
post_title="$1"
command="bundle exec jekyll post \"$post_title\""

# Print the command to be executed and then run it.
echo -e "\n> Executing command:\n  $command\n"
eval "$command"
