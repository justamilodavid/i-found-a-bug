# i-found-a-bug
I found a bug in Basic Algorithm Scripting, truncate a string.  https://learn.freecodecamp.org/javascript-algorithms-and-data-structures/basic-algorithm-scripting/truncate-a-string


Truncate a string (first argument) if it is longer than the given maximum string length (second argument). Return the truncated string with a ... ending.


//this solution cannot pass the challenge//
function truncateString(str, num) {
  // Clear out that junk in your trunk
  if(str.length<=num){
    return str;
  }
  
  var strSliced = str.slice(0,num);
  
  if(strSliced.length<=3){
    return strSliced + "...";
  }
  return strSliced.slice(0,-3) + "...";
}

truncateString("A-tisket a-tasket A green and yellow basket", 11);
