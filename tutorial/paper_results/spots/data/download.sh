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
https://archive.eso.org/downloadportalapi/readme/875431ee-d2f1-4d6b-87b5-d52274d43e7f
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-02T10:02:52.563
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:58.757
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-10T18:32:10.652
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.873
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.511
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.202
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.487
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-01T10:21:32.807
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.905
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.947
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:56.235
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:56.755
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:58.323
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-02T10:02:28.973
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:58.166
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:55.951
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:58.849
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.294
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.099
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.412
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.134
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.893
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-02T10:02:23.310
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.767
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.889
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.447
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.645
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:58.092
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-10T18:32:15.443
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:58.451
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:58.134
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:58.419
https://dataportal.eso.org/dataPortal/file/ADP.2016-09-28T06:54:52.746
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:58.415
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:58.977
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.092
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-10T18:32:11.316
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:56.925
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-01T10:23:26.367
https://dataportal.eso.org/dataPortal/file/ADP.2016-09-21T09:50:22.784
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.102
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.415
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:46:00.315
https://dataportal.eso.org/dataPortal/file/ADP.2016-09-20T06:47:08.769
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:46:00.437
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:56.893
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:56.970
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:56.851
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-10T18:32:11.115
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:59.935
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:58.543
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:46:00.231
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:58.665
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:58.941
https://dataportal.eso.org/dataPortal/file/ADP.2017-05-09T07:46:30.417
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.809
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-10T18:32:15.451
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:46:00.273
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:46:00.060
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-10T18:32:10.645
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T10:12:40.152
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.127
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.169
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.400
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.841
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.445
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-01T10:23:32.803
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.124
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.685
https://dataportal.eso.org/dataPortal/file/ADP.2016-09-27T11:56:46.939
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T10:12:40.117
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-01T10:21:58.207
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:57.717
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:56.489
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:46:00.224
https://dataportal.eso.org/dataPortal/file/ADP.2020-06-19T11:45:56.447
https://dataportal.eso.org/dataPortal/file/ADP.2016-09-27T11:56:46.943
https://dataportal.eso.org/dataPortal/file/ADP.2016-09-27T11:56:49.803
https://dataportal.eso.org/dataPortal/file/ADP.2014-10-02T10:02:24.320
__EOF__
