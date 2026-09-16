import re, sys
s = open(sys.argv[1]).read()
out = []
i = 0
while i < len(s):
    m = re.compile(r'\\rev\{').search(s, i)
    if not m:
        out.append(s[i:]); break
    out.append(s[i:m.start()])
    depth, j = 1, m.end()
    while j < len(s) and depth > 0:
        if s[j] == '{' and s[j-1] != '\\': depth += 1
        elif s[j] == '}' and s[j-1] != '\\': depth -= 1
        j += 1
    out.append(s[m.end():j-1])   # inner content, drop closing brace
    i = j
open(sys.argv[2],'w').write(''.join(out))
