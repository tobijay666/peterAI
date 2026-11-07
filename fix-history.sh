#!/bin/sh
git filter-branch -f --env-filter '
OLD_EMAIL="devmith0702@gmail.com"
CORRECT_NAME="Neola Udeshi Silva"
CORRECT_EMAIL="udeshisilva20@gmail.com"

if [ "" = "" ]
then
    export GIT_COMMITTER_NAME=""
    export GIT_COMMITTER_EMAIL=""
fi
if [ "" = "" ]
then
    export GIT_AUTHOR_NAME=""
    export GIT_AUTHOR_EMAIL=""
fi
' --msg-filter '
sed -e "/Co-authored-by:/d"
' --tag-name-filter cat -- --branches --tags
