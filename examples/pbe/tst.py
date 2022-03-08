import re
from regexp.type_regex import Raw, Match, regex_match

k = Raw("([A-Z])([a-z]+)")
string = "Abc"
print([g for g in re.match("([A-Z])([a-z]+)", string).groups()])
print([g for g in regex_match(k, string).group()])
print([g for g in regex_match(k, string).groups()])
print(regex_match(k, string).group())
