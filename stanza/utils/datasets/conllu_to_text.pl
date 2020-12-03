#!/usr/bin/env perl
# Extracts raw text from CoNLL-U file. Uses newdoc and newpar tags when available.
# Copyright © 2017 Dan Zeman <zeman@ufal.mff.cuni.cz>
# License: GNU GPL

use utf8;
use open ':utf8';
binmode(STDIN, ':utf8');
binmode(STDOUT, ':utf8');
binmode(STDERR, ':utf8');
use Getopt::Long;

# Language code 'zh' or 'ja' will trigger Chinese-like text formatting.
my $language = 'en';
GetOptions
(
    'language=s' => \$language
);
my $chinese = $language =~ m/^(zh|ja|lzh|yue)(_|$)/;

my $text = ''; # from the text attribute of the sentence
my $ftext = ''; # from the word forms of the tokens
my $newpar = 0;
my $newdoc = 0;
my $buffer = '';
my $start = 1;
my $mwtlast;
while(<>)
{
    if(m/^\#\s*text\s*=\s*(.+)/)
    {
        $text = $1;
    }
    elsif(m/^\#\s*newpar(\s|$)/i)
    {
        $newpar = 1;
    }
    elsif(m/^\#\s*newdoc(\s|$)/i)
    {
        $newdoc = 1;
    }
    elsif(m/^\d+-(\d+)\t/)
    {
        $mwtlast = $1;
        my @f = split(/\t/, $_);
        # Paragraphs may start in the middle of a sentence (bulleted lists, verse etc.)
        # The first token of the new paragraph has "NewPar=Yes" in the MISC column.
        # Multi-word tokens have this in the token-introducing line.
        if($f[9] =~ m/NewPar=Yes/i)
        {
            # Empty line between documents and paragraphs. (There may have been
            # a paragraph break before the first part of this sentence as well!)
            $buffer = print_new_paragraph_if_needed($start, $newdoc, $newpar, $buffer);
            $buffer .= $ftext;
            # Line breaks at word boundaries after at most 80 characters.
            $buffer = print_lines_from_buffer($buffer, 80, $chinese);
            print("$buffer\n\n");
            $buffer = '';
            # Start is only true until we write the first sentence of the input stream.
            $start = 0;
            $newdoc = 0;
            $newpar = 0;
            $text = '';
            $ftext = '';
        }
        $ftext .= $f[1];
        $ftext .= ' ' unless($f[9] =~ m/SpaceAfter=No/);
    }
    elsif(m/^(\d+)\t/ && !(defined($mwtlast) && $1<=$mwtlast))
    {
        $mwtlast = undef;
        my @f = split(/\t/, $_);
        # Paragraphs may start in the middle of a sentence (bulleted lists, verse etc.)
        # The first token of the new paragraph has "NewPar=Yes" in the MISC column.
        # Multi-word tokens have this in the token-introducing line.
        if($f[9] =~ m/NewPar=Yes/i)
        {
            # Empty line between documents and paragraphs. (There may have been
            # a paragraph break before the first part of this sentence as well!)
            $buffer = print_new_paragraph_if_needed($start, $newdoc, $newpar, $buffer);
            $buffer .= $ftext;
            # Line breaks at word boundaries after at most 80 characters.
            $buffer = print_lines_from_buffer($buffer, 80, $chinese);
            print("$buffer\n\n");
            $buffer = '';
            # Start is only true until we write the first sentence of the input stream.
            $start = 0;
            $newdoc = 0;
            $newpar = 0;
            $text = '';
            $ftext = '';
        }
        $ftext .= $f[1];
        $ftext .= ' ' unless($f[9] =~ m/SpaceAfter=No/);
    }
    elsif(m/^\s*$/)
    {
        # In a valid CoNLL-U file, $text should be equal to $ftext except for the
        # space after the last token. However, if there have been intra-sentential
        # paragraph breaks, $ftext contains only the part after the last such
        # break, and $text is empty. Hence we currently use $ftext everywhere
        # and ignore $text, even though we note it when seeing the text attribute.
        # $text .= ' ' unless($chinese);
        # Empty line between documents and paragraphs.
        $buffer = print_new_paragraph_if_needed($start, $newdoc, $newpar, $buffer);
        $buffer .= $ftext;
        # Line breaks at word boundaries after at most 80 characters.
        $buffer = print_lines_from_buffer($buffer, 80, $chinese);
        # Start is only true until we write the first sentence of the input stream.
        $start = 0;
        $newdoc = 0;
        $newpar = 0;
        $text = '';
        $ftext = '';
        $mwtlast = undef;
    }
}
# There may be unflushed buffer contents after the last sentence, less than 80 characters
# (otherwise we would have already dealt with it), so just flush it.
if($buffer ne '')
{
    print("$buffer\n");
}



#------------------------------------------------------------------------------
# Checks whether we have to print an extra line to separate paragraphs. Does it
# if necessary. Returns the updated buffer.
#------------------------------------------------------------------------------
sub print_new_paragraph_if_needed
{
    my $start = shift;
    my $newdoc = shift;
    my $newpar = shift;
    my $buffer = shift;
    if(!$start && ($newdoc || $newpar))
    {
        if($buffer ne '')
        {
            print("$buffer\n");
            $buffer = '';
        }
        print("\n");
    }
    return $buffer;
}



#------------------------------------------------------------------------------
# Prints as many complete lines of text as there are in the buffer. Returns the
# remaining contents of the buffer.
#------------------------------------------------------------------------------
sub print_lines_from_buffer
{
    my $buffer = shift;
    # Maximum number of characters allowed on one line, not counting the line
    # break character(s), which also replace any number of trailing spaces.
    # Exception: If there is a word longer than the limit, it will be printed
    # on one line.
    # Note that this algorithm is not suitable for Chinese and Japanese.
    my $limit = shift;
    # We need a different algorithm for Chinese and Japanese.
    my $chinese = shift;
    if($chinese)
    {
        return print_chinese_lines_from_buffer($buffer, $limit);
    }
    if(length($buffer) >= $limit)
    {
        my @cbuffer = split(//, $buffer);
        # There may be more than one new line waiting in the buffer.
        while(scalar(@cbuffer) >= $limit)
        {
            ###!!! We could make it simpler if we ignored multi-space sequences
            ###!!! between words. It sounds OK to ignore them because at the
            ###!!! line break we do not respect original spacing anyway.
            my $i;
            my $ilastspace;
            for($i = 0; $i<=$#cbuffer; $i++)
            {
                if($i>$limit && defined($ilastspace))
                {
                    last;
                }
                if($cbuffer[$i] =~ m/\s/)
                {
                    $ilastspace = $i;
                }
            }
            if(defined($ilastspace) && $ilastspace>0)
            {
                my @out = @cbuffer[0..($ilastspace-1)];
                splice(@cbuffer, 0, $ilastspace+1);
                print(join('', @out), "\n");
            }
            else
            {
                print(join('', @cbuffer), "\n");
                splice(@cbuffer);
            }
        }
        $buffer = join('', @cbuffer);
    }
    return $buffer;
}



#------------------------------------------------------------------------------
# Prints as many complete lines of text as there are in the buffer. Returns the
# remaining contents of the buffer. Assumes that there are no spaces between
# words and lines can be broken between any two characters, as is the custom in
# Chinese and Japanese.
#------------------------------------------------------------------------------
sub print_chinese_lines_from_buffer
{
    my $buffer = shift;
    # Maximum number of characters allowed on one line, not counting the line
    # break character(s).
    my $limit = shift;
    # We cannot simply print the first $limit characters from the buffer,
    # followed by a line break. There could be embedded Latin words or
    # numbers and we do not want to insert a line break in the middle of
    # a foreign word.
    my @cbuffer = split(//, $buffer);
    while(scalar(@cbuffer) >= $limit)
    {
        my $nprint = 0;
        for(my $i = 0; $i <= $#cbuffer; $i++)
        {
            if($i > $limit && $nprint > 0)
            {
                last;
            }
            unless($i < $#cbuffer && $cbuffer[$i] =~ m/[\p{Latin}0-9]/ && $cbuffer[$i+1] =~ m/[\p{Latin}0-9]/)
            {
                $nprint = $i+1;
            }
        }
        my @out = @cbuffer[0..($nprint-1)];
        splice(@cbuffer, 0, $nprint);
        print(join('', @out), "\n");
    }
    $buffer = join('', @cbuffer);
    return $buffer;
}
