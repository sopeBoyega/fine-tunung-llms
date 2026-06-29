# JSX Style Fine-Tune: Base vs Fine-Tuned Comparison

Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
Adapter: LoRA, r=16, alpha=32, ~21 training examples (5 real + synthetic), 8 epochs

Note: temperature=0.2 sampling, so re-running produces slightly different
text each time. Judge the overall pattern across prompts, not one completion.

## Style Pattern Score Summary

Out of 6 held-out prompts, how many completions matched each style pattern:

| Pattern | Base model | Fine-tuned |
|---|---|---|
| tailwind_arbitrary_values | 0/6 | 3/6 |
| typed_props_destructured | 0/6 | 2/6 |
| default_export_at_bottom | 0/6 | 6/6 |
| react_fc_typing | 1/6 | 3/6 |
| union_type_for_constrained_prop | 0/6 | 0/6 |

A higher fine-tuned number than base, on patterns we explicitly trained for,
is evidence the fine-tune actually shifted style — not just a vibe.

---

## Prompt 1

**Prompt:** Write a React component for a 'like' button that shows a heart icon and a count, toggling filled/outline on click.

**Base model style matches:** ['react_fc_typing']
**Base model:**
```
```jsx
import React from 'react';
import { useState } from 'react';

// Define the props interface
interface LikeButtonProps {
  onClick: () => void;
}

const LikeButton: React.FC<LikeButtonProps> = ({ onClick }) => {
  // State to track if the like button is active
  const [isLiked, setIsLiked] = useState(false);

  return (
    <div className="flex items-center space-x-2">
      {/* Heart icon */}
      <svg
        xmlns="http://www.w3.org/2000/svg"
        fill={isLiked ? 'red' : 'none'}
        viewBox="0 0 24 24"
        stroke-width="2"
        stroke="currentColor"
        className={`w-6 h-6 ${isLiked ? 'fill-red' : ''}`}
      >
        <path d="M21.59 17.18L14.59 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1-2 2v14h14a2 2 0 0 1 2 2v14h14a2 2 0 0 1 2 2v14h14a2 2 0 0 1 2 2v14h14a2 2 0 0 1 2 2v14h14a2 2 0 0 1 2 2v14h14a2 2 0 0 1 2 2v14h14a2 2 0 0 1 2 2v14h14a2 2 0 0 1 2 2v14h
```

**Fine-tuned style matches:** ['tailwind_arbitrary_values', 'typed_props_destructured', 'default_export_at_bottom']
**Fine-tuned model:**
```
import React, { useState } from "react";

type LikeButtonProps = {
  initialCount?: number;
};

const LikeButton = ({ initialCount = 0 }: LikeButtonProps) => {
  const [count, setCount] = useState<number>(initialCount);

  const handleLikeClick = () => {
    setCount(count + 1);
  };

  return (
    <div className="flex items-center gap-2">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        strokeWidth={1.5}
        stroke="currentColor"
        className="w-[18px] h-[18px] text-gray-700"
      >
        <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12h18m-9.75 3H5.25A2.25 2.25 0 013 10.125V7.5a2.25 2.25 0 012-2v7.5A2.25 2.25 0 0116.5 21h3m-3-9l3 9-3 9" />
      </svg>
      <span className="text-sm font-medium text-gray-700">{count}</span>
    </div>
  );
};

export default LikeButton;
```

---

## Prompt 2

**Prompt:** Write a React/Next.js component for a comment box with a textarea and a submit button.

**Base model style matches:** []
**Base model:**
```
```jsx
import React from 'react';

// Define the CommentBox component
const CommentBox = ({ onCommentSubmit }) => {
  // State to hold the comment text
  const [commentText, setCommentText] = React.useState('');

  // Function to handle form submission
  const handleSubmit = (event) => {
    event.preventDefault();
    if (commentText.trim() !== '') {
      onCommentSubmit(commentText);
      setCommentText('');
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      {/* Textarea for users to enter their comments */}
      <textarea
        value={commentText}
        onChange={(e) => setCommentText(e.target.value)}
        placeholder="Type your comment here..."
        rows={3}
        className="w-full border border-gray-400 rounded-lg py-2 px-4 focus:outline-none"
      />
      {/* Submit button to send the comment */}
      <button
        type="submit"
        onClick={handleSubmit}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg mt-4"
      >
        Submit
      </button>
    </div>
  );
};

// Default export of the CommentBox component
export default CommentBox;
```

This code snippet defines a functional component named `CommentBox` which takes one prop `onCommentSubmit`, which is a function that will be called when the user submits their comment. The component includes a textarea where users can input their comments and a submit button that sends the comment to the specified handler. The component uses Tailwind CSS for styling and has a simple structure with clear comments explaining each part of the code.
```

**Fine-tuned style matches:** ['tailwind_arbitrary_values', 'default_export_at_bottom', 'react_fc_typing']
**Fine-tuned model:**
```
import React, { useState } from 'react';

type CommentBoxProps = {
  onSubmit: (commentText: string) => void;
};

const CommentBox: React.FC<CommentBoxProps> = ({ onSubmit }) => {
  const [commentText, setCommentText] = useState('');

  const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCommentText(event.target.value);
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (commentText.trim() === '') return;

    onSubmit(commentText);
    setCommentText('');
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2 w-full">
      <textarea
        value={commentText}
        onChange={handleInputChange}
        placeholder="Write a comment..."
        rows={3}
        className="w-full h-[40px] border-b-[1px] border-gray-300 rounded-tl-none rounded-tr-none resize-none focus:outline-none focus:border-blue-500"
      />
      <button type="submit" className="py-[8px] px-[16px] bg-blue-500 text-white rounded-bl-none rounded-br-none hover:bg-blue-700 transition-colors duration-200">
        Post
      </button>
    </form>
  );
};

export default CommentBox;
```

---

## Prompt 3

**Prompt:** Write a React component for a progress bar that fills based on a percentage prop.

**Base model style matches:** []
**Base model:**
```
```jsx
// Import necessary dependencies
import { useState } from 'react';

// Define the Progress Bar component
const ProgressBar = ({ percentage }) => {
  // State to track the progress of the bar
  const [progress, setProgress] = useState(0);

  // Function to update the progress based on the percentage prop
  const handlePercentageChange = (newPercentage) => {
    setProgress(newPercentage);
  };

  return (
    <div className="relative w-full h-4 bg-gray-200 rounded-lg">
      {/* Inner div to fill the progress */}
      <div
        className={`w-${percentage}% bg-blue-500 rounded-lg`}
        role="progressbar"
        aria-valuenow={percentage}
        aria-valuemin={0}
        aria-valuemax={100}
      />
    </div>
  );
};

// Default export of the component
export default ProgressBar;
```

This code snippet defines a `ProgressBar` component that takes a `percentage` prop and displays a progress bar filled according to that percentage. The progress bar is styled using Tailwind CSS classes, and the inner div (`<div>` with the `bg-blue-500` class) fills the specified percentage of the parent container. The `handlePercentageChange` function allows external components to update the progress dynamically.
```

**Fine-tuned style matches:** ['default_export_at_bottom', 'react_fc_typing']
**Fine-tuned model:**
```
import React from 'react';

type ProgressBarProps = {
  percentComplete: number;
};

const ProgressBar: React.FC<ProgressBarProps> = ({ percentComplete }) => {
  const progressBarStyle = `w-full h-4 bg-gray-200 rounded-full overflow-hidden`;
  const fillWidth = `${percentComplete}%`;

  return (
    <div className={progressBarStyle}>
      <div
        className={`bg-blue-500 h-full w-${fillWidth}`}
        role="progressbar"
        aria-valuenow={percentComplete}
        aria-valuemin={0}
        aria-valuemax={100}
      />
    </div>
  );
};

export default ProgressBar;
```

---

## Prompt 4

**Prompt:** Write a React component for a navbar with a logo on the left and nav links on the right.

**Base model style matches:** []
**Base model:**
```
```jsx
import React from 'react';
import { Link } from 'next/link';

const Navbar = ({ siteTitle }) => {
  return (
    <nav className="bg-gray-800">
      <div className="container mx-auto flex justify-between items-center py-4">
        <Link href="/" className="text-white font-bold text-xl">
          {siteTitle}
        </Link>
        <ul className="flex space-x-2">
          <li>
            <Link href="/about" className="text-white hover:text-blue-500">About</Link>
          </li>
          <li>
            <Link href="/services" className="text-white hover:text-blue-500">Services</Link>
          </li>
          <li>
            <Link href="/contact" className="text-white hover:text-blue-500">Contact</Link>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
```

This code snippet defines a functional component named `Navbar` which takes one prop `siteTitle`. The component renders a navigation bar with a logo on the left and three nav links on the right. Each link is styled to be white and change color on hover to blue. The `Link` component from Next.js is used to create the clickable links. The `navbar` class is defined in a Tailwind CSS file to apply styles such as background color, container width, alignment, padding, and spacing.
```

**Fine-tuned style matches:** ['default_export_at_bottom']
**Fine-tuned model:**
```
import React from 'react'
import Link from 'next/link'

const Navbar = () => {
  return (
    <nav className="flex items-center justify-between bg-white px-6 py-3 shadow-md">
      <Link href="/">
        <a className="text-blue-700 font-bold text-xl">My Website</a>
      </Link>

      <div className="ml-auto flex items-center">
        <ul className="flex space-x-4">
          <li>
            <Link href="/about">
              <a className="hover:text-blue-800 transition-colors duration-200">About</a>
            </Link>
          </li>
          <li>
            <Link href="/services">
              <a className="hover:text-blue-800 transition-colors duration-200">Services</a>
            </Link>
          </li>
          <li>
            <Link href="/contact">
              <a className="hover:text-blue-800 transition-colors duration-200">Contact</a>
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  )
}

export default Navbar
```

---

## Prompt 5

**Prompt:** Write a React component for an empty-state message shown when a list has no items, with an icon and a short message.

**Base model style matches:** []
**Base model:**
```
```jsx
import { useState } from 'react';
import { CheckIcon } from '@heroicons/react/24/solid';

const EmptyState = ({ message }) => {
  return (
    <div className="flex flex-col justify-center items-center h-screen">
      <CheckIcon className="w-16 h-16 text-green-500" />
      <p className="text-lg font-bold">{message}</p>
    </div>
  );
};

export default EmptyState;
```

In this solution:
- The component is named `EmptyState`.
- It takes one prop: `message`, which should be a string.
- The component uses Tailwind CSS classes to style the icon and the message.
- The icon is displayed using the `CheckIcon` component from `@heroicons/react/24/solid`.
- The message is displayed as a paragraph (`<p>`) styled with `font-size: 2xl;`.
- The component is exported by default at the end of the file.
```

**Fine-tuned style matches:** ['typed_props_destructured', 'default_export_at_bottom']
**Fine-tuned model:**
```
import React from 'react';
import { CheckCircleIcon } from '@heroicons/react/24/solid';

type EmptyStateProps = {
  heading?: string;
  subheading?: string;
};

const EmptyState = ({ heading = 'No items', subheading }: EmptyStateProps) => {
  return (
    <div className="flex flex-col items-center justify-center h-full">
      <CheckCircleIcon className="text-[#6B7280] w-16 h-16" />
      <h3 className="text-[#525252] text-xl font-medium mt-2">{heading}</h3>
      <p className="text-[#718093] text-base mt-1">{subheading}</p>
    </div>
  );
};

export default EmptyState;
```

---

## Prompt 6

**Prompt:** Write a React component for a badge/label that shows one of three priority levels: low, medium, high.

**Base model style matches:** []
**Base model:**
```
```jsx
import { useState } from 'react';

const PriorityBadge = ({ priority }) => {
  const [selectedPriority, setSelectedPriority] = useState(priority);

  return (
    <div className="bg-blue-500 text-white rounded-full px-2 py-1">
      {selectedPriority}
    </div>
  );
};

export default PriorityBadge;
```

This component uses React hooks to manage the state of the selected priority level. The `priority` prop is destructured from the props object and passed to the component. The component renders a badge with the selected priority level.
```

**Fine-tuned style matches:** ['tailwind_arbitrary_values', 'default_export_at_bottom', 'react_fc_typing']
**Fine-tuned model:**
```
import React from 'react';

type PriorityBadgeProps = {
  priority: 'low' | 'medium' | 'high';
};

const PriorityBadge: React.FC<PriorityBadgeProps> = ({ priority }) => {
  const colorMap: Record<string, string> = {
    low: 'bg-blue-500',
    medium: 'bg-yellow-500',
    high: 'bg-red-500',
  };

  return (
    <div className={`flex items-center px-[8px] py-[4px] rounded-full ${colorMap[priority]}`}>
      {priority.toUpperCase()}
    </div>
  );
};

export default PriorityBadge;
```

---
