#!/bin/sh

usage () {
    cat <<__EOF__
usage: $(basename "$0") [-hlp] [-u user] [-X args] [-d args]
  -h        print this help text
  -l        print list of files to download
  -p        prompt for password
  -u user   download as a different user
  -X args   extra arguments to pass to xargs
  -d args   extra arguments to pass to the download program

__EOF__
}

hostname=dataportal.eso.org
username=anonymous
anonymous=
xargsopts=
prompt=
list=
while getopts hlpu:xX:d: option
do
    case $option in
	h) usage; exit ;;
	l) list=yes ;;
	p) prompt=yes ;;
	u) prompt=yes; username="$OPTARG" ;;
	X) xargsopts="$OPTARG" ;;
	d) download_opts="$OPTARG";;
	?) usage; exit 2 ;;
    esac
done

if [ "$username" = "anonymous" ]; then
    anonymous=yes
fi

if [ -z "$xargsopts" ]; then
    #no xargs option specified, we ensure that only one url
    #after the other will be used
    xargsopts='-L 1'
fi

netrc=$HOME/.netrc
if [ -z "$anonymous" ] && [ -z "$prompt" ]; then
    # take password (and user) from netrc if no -p option
    if [ -f "$netrc" ] && [ -r "$netrc" ]; then
	grep -ir "$hostname" "$netrc" > /dev/null
	if [ $? -ne 0 ]; then
            #no entry for $hostname, user is prompted for password
            echo "A .netrc is available but there is no entry for $hostname, add an entry as follows if you want to use it:"
            echo "machine $hostname login anonymous password _yourpassword_"
            prompt="yes"
	fi
    else
	prompt="yes"
    fi
fi

if [ -n "$prompt" ] && [ -z "$list" ]; then
    trap 'stty echo 2>/dev/null; echo "Cancelled."; exit 1' INT HUP TERM
    stty -echo 2>/dev/null
    printf 'Password: '
    read password
    echo ''
    stty echo 2>/dev/null
    escaped_password=${password//\%/\%25}
    auth_check=$(wget -O - --post-data "username=$username&password=$escaped_password" --server-response --no-check-certificate "https://www.eso.org/sso/oidc/accessToken?grant_type=password&client_id=clientid" 2>&1 | awk '/^  HTTP/{print $2}')
    if [ ! "$auth_check" -eq 200 ]
    then
        echo 'Invalid password!'
        exit 1
    fi
fi

# use a tempfile to which only user has access 
tempfile=$(mktemp /tmp/dl.XXXXXXXX 2>/dev/null)
test "$tempfile" -a -f "$tempfile" || {
    tempfile=/tmp/dl.$$
    ( umask 077 && : >$tempfile )
}
trap 'rm -f $tempfile' EXIT INT HUP TERM

echo "auth_no_challenge=on" > "$tempfile"
# older OSs do not seem to include the required CA certificates for ESO
echo "check_certificate=off" >> "$tempfile"
echo "content_disposition=on" >> "$tempfile"
if [ -z "$anonymous" ] && [ -n "$prompt" ]; then
    echo "http_user=$username" >> "$tempfile"
    echo "http_password=$password" >> "$tempfile"
fi
WGETRC=$tempfile; export WGETRC

unset password

if [ -n "$list" ]; then
    cat
else
    xargs "$xargsopts" wget "$download_opts"
fi <<'__EOF__'
https://archive.eso.org/downloadportalapi/readme/51c81b5e-30c6-432c-bf6a-7f76f649d3b5
https://dataportal.eso.org/dataPortal/file/ADP.2014-09-26T16:54:25.233
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-07T13:37:24.093
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-06T10:07:43.360
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-07T08:33:58.797
https://dataportal.eso.org/dataPortal/file/ADP.2014-09-23T11:02:21.517
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-07T08:34:07.833
https://dataportal.eso.org/dataPortal/file/ADP.2014-09-24T09:41:41.830
https://dataportal.eso.org/dataPortal/file/ADP.2014-09-26T16:53:04.683
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-06T10:05:28.520
https://dataportal.eso.org/dataPortal/file/ADP.2014-09-24T09:41:33.393
https://dataportal.eso.org/dataPortal/file/ADP.2014-09-26T16:52:30.377
https://dataportal.eso.org/dataPortal/file/ADP.2014-09-24T09:44:27.687
https://dataportal.eso.org/dataPortal/file/ADP.2014-09-26T16:51:02.083
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-07T08:33:44.830
https://dataportal.eso.org/dataPortal/file/ADP.2014-12-14T22:55:26.757
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-07T08:33:57.643
https://dataportal.eso.org/dataPortal/file/ADP.2015-01-07T22:55:22.590
https://dataportal.eso.org/dataPortal/file/ADP.2014-09-23T11:00:36.353
https://dataportal.eso.org/dataPortal/file/ADP.2014-09-24T09:44:46.297
https://dataportal.eso.org/dataPortal/file/ADP.2014-09-23T11:03:55.627
https://dataportal.eso.org/dataPortal/file/ADP.2014-09-24T09:41:24.623
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-06T10:04:53.960
__EOF__
